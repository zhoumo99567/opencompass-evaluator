import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import subprocess

import code_runtime as runtime


ROOT = Path(__file__).resolve().parents[1]
IMAGE_ID = 'sha256:' + 'a' * 64


class CodeRuntimeTests(unittest.TestCase):
    def tearDown(self):
        runtime._CACHE.clear()

    def test_fixed_container_boundary_has_no_host_mount_or_credentials(self):
        args = runtime.container_command('fixed-name', IMAGE_ID)
        for flag, value in (('--network', 'none'), ('--user', '65534:65534'),
                            ('--cap-drop', 'ALL'), ('--memory', '256m'), ('--pids-limit', '32')):
            self.assertEqual(args[args.index(flag) + 1], value)
        self.assertIn('--read-only', args)
        self.assertIn('no-new-privileges', args)
        self.assertEqual(args[-1], IMAGE_ID)
        for forbidden in ('--privileged', '-v', '--volume', '--mount', '-e', '--env', '--env-file'):
            self.assertNotIn(forbidden, args)
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'do-not-copy', 'SECRET_TOKEN': 'do-not-copy'}):
            env = runtime.clean_environment()
            self.assertNotIn('OPENAI_API_KEY', env)
            self.assertNotIn('SECRET_TOKEN', env)

    def test_proof_is_bound_to_live_image_and_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            digest = 'test-source'
            state = dict(docker_available=True, daemon_available=True, image_available=True,
                         image=runtime.IMAGE, image_id=IMAGE_ID, protocol='1', source_hash=digest, reason='')
            runtime.proof_path(root).parent.mkdir()
            runtime.proof_path(root).write_text(json.dumps(dict(ready=True, protocol=1,
                image_id=IMAGE_ID, source_hash=digest, supported_datasets=list(runtime.DATASETS))))
            with patch.object(runtime, 'runtime_status', return_value=state), \
                    patch.object(runtime, 'source_hash', return_value=digest):
                self.assertTrue(runtime.environment_status(root, refresh=True)['ready'])
                with patch.object(runtime, 'source_hash', return_value='changed-source'):
                    self.assertFalse(runtime.environment_status(root, refresh=True)['ready'])
                with patch.object(runtime, 'runtime_status', return_value=dict(state, image_id='different')):
                    self.assertFalse(runtime.environment_status(root, refresh=True)['ready'])
                with patch.object(runtime, 'runtime_status', return_value=dict(state, image_available=False, daemon_available=False)):
                    self.assertFalse(runtime.environment_status(root, refresh=True)['ready'])

    def test_self_test_must_pass_and_match_nonroot_protocol_before_proof(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = dict(image_available=True, image_id=IMAGE_ID, protocol='1', source_hash='digest', reason='')
            checks = {key: True for key in ('pass', 'wrong_answer', 'timeout', 'early_exit', 'stdout_isolated')}
            response = dict(protocol=1, passed=True, checks=checks, uid=65534, supported=['humaneval_cn', 'mbpp'])
            with patch.object(runtime, 'runtime_status', return_value=state), \
                    patch.object(runtime, 'source_hash', return_value='digest'), \
                    patch.object(runtime, 'environment_status'), \
                    patch.object(runtime, '_run_container', return_value=response) as run:
                proof = runtime.verify_environment(root)
                self.assertTrue(proof['ready'])
                self.assertEqual(proof['image_id'], IMAGE_ID)
                self.assertEqual(run.call_args.kwargs['image_id'], IMAGE_ID)
                self.assertEqual(json.loads(runtime.proof_path(root).read_text())['checks'], checks)
                self.assertEqual(runtime.proof_path(root).stat().st_mode & 0o777, 0o600)
                for bad in (dict(response, uid=0), dict(response, passed=False),
                            dict(response, supported=['mbpp'])):
                    run.return_value = bad
                    with self.assertRaises(runtime.CodeEnvironmentError):
                        runtime.verify_environment(root)

    def test_each_case_uses_verified_immutable_image(self):
        outcome = dict(protocol=1, passed=False, result='wrong_answer')
        with patch.object(runtime, 'environment_status', return_value=dict(ready=True, image_id=IMAGE_ID)), \
                patch.object(runtime, '_run_container', return_value=outcome) as run:
            self.assertEqual(runtime.execute_case(ROOT, 'mbpp', 'candidate', {'tests': 'assert True'}),
                             dict(outcome, image_id=IMAGE_ID))
            self.assertEqual(run.call_args.kwargs['image_id'], IMAGE_ID)
            self.assertEqual(run.call_args.args[1]['completion'], 'candidate')
        with patch.object(runtime, 'environment_status', return_value=dict(ready=False, reason='stopped')):
            with self.assertRaisesRegex(runtime.CodeEnvironmentError, 'stopped'):
                runtime.execute_case(ROOT, 'mbpp', 'candidate', {'tests': 'assert True'})

    def test_timeout_cleans_only_the_named_container(self):
        process = Mock(stdin=io.BytesIO(), stdout=io.BytesIO(), stderr=io.BytesIO(), returncode=-9)
        process.wait.side_effect = [subprocess.TimeoutExpired('docker', 1), -9]
        with patch.object(runtime.subprocess, 'Popen', return_value=process) as popen, \
                patch.object(runtime, '_control') as control:
            with self.assertRaises(runtime.CodeEnvironmentError):
                runtime._run_container([], {'protocol': 1}, timeout=1)
            command = popen.call_args.args[0]
            container_name = command[command.index('--name') + 1]
            control.assert_called_once_with(['docker', 'rm', '-f', container_name], timeout=5)
            process.kill.assert_called_once()


if __name__ == '__main__':
    unittest.main()
