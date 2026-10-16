import copy
import io
import json
import os
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from environment_manager import EnvironmentManager, InstallationError, MAX_LOGS, clean_subprocess_env


class RuntimeFixture:
    DATASETS = {'humaneval_cn_gen': {}, 'mbpp_gen': {}}

    def __init__(self):
        self.state = {'docker_available': True, 'daemon_available': True, 'image_available': True,
                      'image': 'local/opencompass-code-eval:1', 'image_id': 'sha256:fixture',
                      'ready': False, 'self_test_passed': False, 'supported_datasets': [], 'reason': '尚未自检'}
        self.data_ready = False
        self.calls = []

    def runtime_status(self, root):
        self.calls.append('runtime_status')
        return copy.deepcopy(self.state)

    def environment_status(self, root):
        return copy.deepcopy(self.state)

    def build_command(self, root):
        return ['/fixed/docker', 'build', '--tag', self.state['image'], str(Path(root) / 'code_sandbox')]

    def verify_environment(self, root):
        self.calls.append('verify')
        self.state.update(ready=True, self_test_passed=True, supported_datasets=list(self.DATASETS))
        return {'ready': True, 'protocol': 1, 'image_id': 'sha256:fixture', 'supported_datasets': list(self.DATASETS)}

    def data_status(self, root):
        return [{'id': dataset_id, 'label': dataset_id, 'ready': self.data_ready,
                 'reason': '' if self.data_ready else '固定数据未准备'} for dataset_id in self.DATASETS]


class EnvironmentManagerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / 'code_sandbox').mkdir()
        for name in ('Dockerfile', 'runner.py'):
            (self.root / 'code_sandbox' / name).write_text('fixture')
        self.runtime = RuntimeFixture()
        self.runtime_patch = patch('environment_manager._runtime_api', return_value=self.runtime)
        self.runtime_patch.start()
        self.manager = EnvironmentManager(self.root)

    def tearDown(self):
        if self.manager.process_lock:
            self.manager._release_process_lock(self.manager.process_lock)
            self.manager.process_lock = None
        self.runtime_patch.stop()
        self.tmp.cleanup()

    def wait_finished(self, manager=None):
        manager = manager or self.manager
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            state = manager.status()
            if not state['busy']:
                return state
            time.sleep(.01)
        self.fail('fixture installation did not finish')

    def downloader(self, root, progress):
        self.runtime.calls.append('data')
        progress('固定数据下载与校验完成')
        self.runtime.data_ready = True
        return {'humaneval_cn': {'ready': True}, 'mbpp': {'ready': True}}

    def test_fixed_install_pipeline_status_and_atomic_private_state(self):
        data = Mock(prepare_code_data=self.downloader)
        with patch.object(self.manager, '_run_command') as command, patch('environment_manager._data_api', return_value=data):
            self.manager.start('install_code')
            final = self.wait_finished()
        self.assertEqual(final['status'], 'completed')
        self.assertEqual([step['id'] for step in final['steps']], ['runtime', 'build', 'self_test', 'data'])
        self.assertTrue(all(step['status'] == 'completed' for step in final['steps']))
        self.assertTrue(final['code']['ready'])
        self.assertTrue(final['code']['self_test_passed'])
        self.assertTrue(all(row['ready'] for row in final['code']['data']))
        self.assertLess(self.runtime.calls.index('verify'), self.runtime.calls.index('data'))
        command.assert_called_once_with(self.runtime.build_command(self.manager.root), 900, '构建代码评测镜像')
        self.assertEqual(self.manager.state_path.stat().st_mode & 0o777, 0o600)
        self.assertEqual(json.loads(self.manager.state_path.read_text())['status'], 'completed')
        self.assertFalse(list(self.manager.state_path.parent.glob('*.tmp')))

    def test_read_only_status_never_attests_ready_from_completed_job_alone(self):
        with self.manager.lock:
            self.manager.job.update(status='completed', action='install_code')
            self.manager._save()
        state = self.manager.status()
        self.assertFalse(state['code']['ready'])
        self.assertFalse(state['code']['self_test_passed'])
        self.assertEqual(self.runtime.calls, [])
        for key in ('status', 'action', 'step', 'steps', 'logs', 'error', 'runtime', 'code', 'can_install', 'can_start_runtime', 'busy'):
            self.assertIn(key, state)

    def test_unknown_actions_and_cross_service_lock_cannot_launch_more_jobs(self):
        for action in (None, [], 'install_code; echo unsafe', 'pip_install', 'http://example.invalid'):
            with self.subTest(action=action), self.assertRaises(ValueError):
                self.manager.start(action)
        with patch('environment_manager.threading.Thread'):
            self.manager.start('check')
        with self.assertRaisesRegex(ValueError, '正在运行'):
            self.manager.start('install_code')
        other = EnvironmentManager(self.root)
        self.assertTrue(other.status()['busy'])
        with self.assertRaisesRegex(ValueError, '另一个服务'):
            other.start('install_code')
        other.recover_interrupted()
        self.assertEqual(json.loads(self.manager.state_path.read_text())['status'], 'running')

    def test_failure_is_retryable_and_bad_self_test_never_marks_data_available(self):
        data = Mock(prepare_code_data=self.downloader)
        with patch.object(self.manager, '_run_command', side_effect=InstallationError('固定构建失败')):
            self.manager.start('install_code')
            failed = self.wait_finished()
        self.assertEqual(failed['status'], 'failed')
        self.assertEqual(failed['steps'][1]['status'], 'failed')
        self.assertFalse(failed['code']['ready'])
        first_id = failed['id']
        with patch.object(self.manager, '_run_command'), patch('environment_manager._data_api', return_value=data):
            self.manager.start('install_code')
            fixed = self.wait_finished()
        self.assertEqual(fixed['status'], 'completed')
        self.assertNotEqual(fixed['id'], first_id)
        self.runtime.state.update(ready=False, self_test_passed=False)
        with patch.object(self.runtime, 'verify_environment', return_value={'ready': True, 'protocol': 2}), \
             patch.object(self.manager, '_run_command'), patch('environment_manager._data_api') as downloads:
            self.manager.start('install_code')
            failed = self.wait_finished()
        self.assertEqual(failed['status'], 'failed')
        self.assertFalse(failed['code']['ready'])
        downloads.assert_not_called()

    def test_missing_cn_data_preserves_usable_sandbox_and_mbpp_with_explicit_warning(self):
        result = {'humaneval_cn': {'ready': False, 'reason': '没有已核实的自动下载源，请准备中文数据'},
                  'mbpp': {'ready': True}}
        rows = [{'id': 'humaneval_cn_gen', 'label': 'HumanEval CN', 'ready': False, 'reason': result['humaneval_cn']['reason']},
                {'id': 'mbpp_gen', 'label': 'MBPP', 'ready': True, 'reason': ''}]
        with patch.object(self.manager, '_run_command'), \
             patch('environment_manager._data_api', return_value=Mock(prepare_code_data=Mock(return_value=result))), \
             patch.object(self.runtime, 'data_status', return_value=rows):
            self.manager.start('install_code')
            state = self.wait_finished()
        self.assertEqual(state['status'], 'completed')
        self.assertTrue(state['code']['ready'])
        self.assertEqual(state['steps'][-1]['status'], 'partial')
        self.assertIsNone(state['error'])
        self.assertTrue(state['warnings'])
        data = {item['id']: item for item in state['code']['data']}
        self.assertTrue(data['mbpp_gen']['ready'])
        self.assertFalse(data['humaneval_cn_gen']['ready'])
        self.assertIn('中文数据', data['humaneval_cn_gen']['message'])

    def test_check_runs_self_test_without_build_download_or_starting_runtime(self):
        self.runtime.data_ready = True
        with patch.object(self.manager, '_run_command') as command, patch('environment_manager._data_api') as downloads:
            self.manager.start('check')
            state = self.wait_finished()
        self.assertEqual(state['status'], 'completed')
        self.assertIn('verify', self.runtime.calls)
        command.assert_not_called()
        downloads.assert_not_called()
        self.runtime.state['daemon_available'] = False
        with patch.object(self.manager, '_run_command') as command:
            self.manager.start('check')
            state = self.wait_finished()
        self.assertEqual(state['status'], 'failed')
        command.assert_not_called()

    def test_start_runtime_only_launches_fixed_existing_app_and_does_not_change_context(self):
        self.runtime.state['daemon_available'] = False

        def started(*args):
            self.runtime.state['daemon_available'] = True

        with patch.object(self.manager, '_application_path', return_value='/Applications/OrbStack.app'), \
             patch.object(self.manager, '_run_command', side_effect=started) as command:
            self.manager.start('start_runtime')
            state = self.wait_finished()
        command.assert_called_once_with(['/usr/bin/open', '-g', '-a', '/Applications/OrbStack.app'], 30, '启动容器应用')
        self.assertEqual(state['status'], 'completed')
        self.assertEqual(len(state['steps']), 1)
        self.assertNotIn('verify', self.runtime.calls)
        self.assertFalse(state['code']['ready'])

    def test_interrupted_recovery_keeps_previous_steps_but_not_a_success_claim(self):
        with patch('environment_manager.threading.Thread'):
            self.manager.start('install_code')
        self.manager._step('build', 'running')
        self.manager._release_process_lock(self.manager.process_lock)
        self.manager.process_lock = None
        self.manager.busy = False
        other = EnvironmentManager(self.root)
        other.recover_interrupted()
        status = other.status()
        self.assertEqual(status['status'], 'interrupted')
        self.assertEqual(status['steps'][1]['status'], 'interrupted')
        self.assertFalse(status['code']['ready'])
        self.assertFalse(status['busy'])

    def test_subprocess_whitelist_and_fixed_milestones_do_not_log_secret_output(self):
        process = Mock(pid=12345)
        process.stdout = io.BytesIO(b'OPENAI_API_KEY=SECRET_FIXTURE\n#1 load metadata Bearer SECRET_BEARER\n')
        process.wait.return_value = 0
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'SECRET_FIXTURE', 'CUSTOM_TOKEN': 'HIDDEN',
                                     'AWS_SECRET_ACCESS_KEY': 'CLOUD_SECRET', 'DOCKER_CONTEXT': 'fixture'}), \
             patch('environment_manager.subprocess.Popen', return_value=process) as popen:
            self.manager._run_command(['/fixed/docker', 'build'], 10, '固定构建')
            child_env = popen.call_args.kwargs['env']
            self.assertNotIn('OPENAI_API_KEY', child_env)
            self.assertNotIn('CUSTOM_TOKEN', child_env)
            self.assertNotIn('AWS_SECRET_ACCESS_KEY', child_env)
            self.assertEqual(child_env['DOCKER_CONTEXT'], 'fixture')
            self.assertNotIn('shell', popen.call_args.kwargs)
        saved = self.manager.state_path.read_text()
        for secret in ('SECRET_FIXTURE', 'SECRET_BEARER', 'HIDDEN', 'CLOUD_SECRET'):
            self.assertNotIn(secret, saved)
        self.assertIn('正在检查基础镜像', saved)
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'SECRET_FIXTURE'}):
            self.assertNotIn('OPENAI_API_KEY', clean_subprocess_env())

    def test_subprocess_deadline_terminates_owned_process_group_and_is_retryable(self):
        process = Mock(pid=12345)
        process.stdout = io.BytesIO(b'')
        process.wait.side_effect = [subprocess.TimeoutExpired('fixture', .1), 0]
        with patch('environment_manager.subprocess.Popen', return_value=process), \
             patch('environment_manager.os.killpg') as terminate:
            with self.assertRaisesRegex(InstallationError, '超过等待时限'):
                self.manager._run_command(['/fixed/docker', 'build'], .1, '固定构建')
        terminate.assert_called_once()
        self.assertEqual(terminate.call_args.args[0], 12345)

    def test_logs_are_bounded_and_mask_common_credential_messages(self):
        for index in range(MAX_LOGS + 3):
            self.manager._log(f'固定步骤 {index}')
        self.manager._log('OPENAI_API_KEY=SECRET_FIXTURE Authorization: Bearer SECRET_BEARER https://user:pwd@example.invalid')
        data = json.loads(self.manager.state_path.read_text())
        self.assertEqual(len(data['logs']), MAX_LOGS)
        self.assertNotIn('SECRET_FIXTURE', self.manager.state_path.read_text())
        self.assertNotIn('SECRET_BEARER', self.manager.state_path.read_text())
        self.assertNotIn('user:pwd', self.manager.state_path.read_text())


if __name__ == '__main__':
    unittest.main()
