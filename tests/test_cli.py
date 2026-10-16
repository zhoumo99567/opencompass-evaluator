import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError

import evaluator_cli as cli


class FixtureClient:
    """An in-memory API fixture: no sockets, server process or model calls."""
    def __init__(self, responses):
        self.responses = {key: list(value) for key, value in responses.items()}
        self.calls = []

    def request(self, route, payload=None, timeout=None):
        self.calls.append((route, payload))
        values = self.responses[route]
        value = values.pop(0) if len(values) > 1 else values[0]
        if isinstance(value, Exception):
            raise value
        return value


class CliTests(unittest.TestCase):
    def run_cli(self, argv, responses):
        client = FixtureClient(responses)
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.object(cli, 'ApiClient', return_value=client), \
             patch.object(cli.time, 'sleep'), \
             contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = cli.main(argv)
        return code, stdout.getvalue(), stderr.getvalue(), client

    def test_multi_dataset_wait_returns_exact_run_details_and_clean_machine_json(self):
        started = {'status': 'running', 'log_path': 'outputs/evaluation_fixture.log', 'run_ids': []}
        finished = dict(started, status='completed', run_ids=['run-actual'])
        result = {'id': 'run-actual', 'status': 'completed',
                  'scores': [{'dataset': 'gsm8k', 'metric': 'accuracy', 'score': .5}],
                  'score_interpretations': [{'dataset': 'gsm8k', 'focuses': ['算术推理']}], 'samples': []}
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'result.json'
            code, stdout, stderr, client = self.run_cli([
                'ability', 'run', '--datasets', 'demo_gsm8k_chat_gen', 'demo_math_chat_gen',
                'demo_gsm8k_chat_gen', '--max-out-len', '8192', '--wait', '--json', '--output', str(output),
            ], {'/api/evaluation/start': [started], '/api/evaluation/status': [finished],
                '/api/runs/run-actual': [result]})
            self.assertEqual(code, 0)
            data = json.loads(stdout)
            self.assertEqual(data['runs'], [result])
            self.assertEqual(data['runs'][0]['scores'][0]['score'], .5)
            self.assertEqual(json.loads(output.read_text()), data)
            self.assertEqual(output.stat().st_mode & 0o777, 0o600)
        self.assertEqual(client.calls[0], ('/api/evaluation/start', {
            'datasets': ['demo_gsm8k_chat_gen', 'demo_math_chat_gen'], 'options': {'max_out_len': 8192}}))
        self.assertIn('任务运行中', stderr)
        self.assertNotIn('任务运行中', stdout)
        self.assertFalse(any(route == '/api/settings' for route, _ in client.calls))

    def test_latest_results_chooses_latest_completed_and_keeps_raw_scores_interpretation(self):
        result = {'id': 'complete', 'status': 'completed', 'scores': [{'score': 12.5}],
                  'score_interpretations': [{'focuses': ['长文本检索']}], 'samples': []}
        code, stdout, _, client = self.run_cli(['results', 'latest', '--json'], {
            '/api/runs': [[{'id': 'new-partial', 'status': 'partial'}, {'id': 'complete', 'status': 'completed'},
                           {'id': 'older', 'status': 'completed'}]], '/api/runs/complete': [result]})
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(stdout), result)
        self.assertEqual(client.calls[-1][0], '/api/runs/complete')
        code, _, error, _ = self.run_cli(['results'], {'/api/runs': [[{'id': 'partial', 'status': 'partial'}]]})
        self.assertNotEqual(code, 0)
        self.assertIn('还没有结果完整', error)

    def test_dataset_detail_exposes_purpose_scoring_failure_candidates_and_impacts(self):
        entry = {'id': 'demo', 'label': 'Demo', 'available': True, 'focuses': ['算术推理'],
                 'purpose': '检查数学文字题的多步计算是否正确', 'task_example': '用总价和数量计算单价',
                 'suitable_uses': ['金额问答'], 'metric_explanation': '按标准答案评分', 'low_score_checks': ['核对答案提取'],
                 'impacts': ['金额计算'], 'limitations': '不能替代真实业务测试'}
        code, stdout, _, _ = self.run_cli(['datasets', 'demo'], {'/api/datasets': [[entry]]})
        self.assertEqual(code, 0)
        for content in ('测什么：检查数学文字题的多步计算是否正确', '适合：金额问答', '题型示例（示意）：用总价和数量计算单价', '主要测试：算术推理', '评分含义：按标准答案评分', '核对答案提取', '金额计算', '真实业务测试'):
            self.assertIn(content, stdout)
        code, stdout, _, _ = self.run_cli(['list-datasets', '--available', '--json'], {
            '/api/datasets': [[entry, dict(entry, id='disabled', available=False)]]})
        self.assertEqual(json.loads(stdout), [entry])

    def test_dataset_filters_combine_category_scope_search_and_availability_without_metadata_loss(self):
        entries = [
            {'id': 'math-demo', 'label': 'Small Math', 'purpose': '验证数学流程', 'category_id': 'math',
             'category': '数学', 'scope': 'demo', 'available': True, 'presets': ['quick']},
            {'id': 'math-full', 'label': 'MATH 500', 'purpose': '数学推理能力', 'category_id': 'math',
             'category': '数学', 'scope': 'full', 'available': True, 'presets': ['reasoning']},
            {'id': 'math-disabled', 'label': 'Disabled Math', 'purpose': '数学推理能力', 'category_id': 'math',
             'category': '数学', 'scope': 'full', 'available': False},
            {'id': 'reading-full', 'label': 'Long Reading', 'purpose': '长文本检索', 'category_id': 'long_context',
             'category': '长上下文', 'scope': 'full', 'available': True},
        ]
        code, stdout, stderr, client = self.run_cli([
            'datasets', '--category', 'math', '--scope', 'full', '--search', '推理', '--available', '--json'],
            {'/api/datasets': [entries]})
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(stdout), [entries[1]])
        self.assertEqual(stderr, '')
        self.assertEqual(client.calls, [('/api/datasets', None)])
        for search in ('mAtH 500', 'MATH-FULL', '数学推理'):
            with self.subTest(search=search):
                code, stdout, _, _ = self.run_cli(['datasets', '--available', '--search', search, '--json'],
                                                 {'/api/datasets': [entries]})
                self.assertEqual(json.loads(stdout), [entries[1]])
        code, stdout, _, _ = self.run_cli(['datasets', '--category', 'math'], {'/api/datasets': [entries]})
        self.assertEqual(code, 0)
        self.assertIn('[数学 (math) / full]', stdout)

    def test_dataset_single_item_filters_cannot_silently_select_another_dataset(self):
        entries = [{'id': 'demo', 'label': 'Demo', 'purpose': 'demo', 'category_id': 'math',
                    'category': '数学', 'scope': 'demo', 'available': True}]
        for filters in (['--scope', 'full'], ['--category', 'knowledge'], ['--search', 'missing']):
            with self.subTest(filters=filters):
                code, stdout, stderr, _ = self.run_cli(['datasets', 'demo', '--json'] + filters,
                                                       {'/api/datasets': [entries]})
                self.assertEqual(code, 2)
                self.assertEqual(stdout, '')
                self.assertIn('不符合', stderr)
        code, stdout, _, _ = self.run_cli(['datasets', 'demo', '--scope', 'demo', '--json'],
                                         {'/api/datasets': [entries]})
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(stdout), entries[0])
        code, stdout, stderr, _ = self.run_cli(['datasets', '--category', 'nonexistent', '--json'],
                                              {'/api/datasets': [entries]})
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(stdout), [])
        self.assertIn('没有符合', stderr)

    def test_failed_partial_and_cancelled_terminal_states_return_nonzero(self):
        for status in ('failed', 'cancelled'):
            with self.subTest(status=status):
                started = {'status': 'running', 'log_path': 'fixture', 'run_ids': []}
                code, stdout, _, _ = self.run_cli(['ability', 'run', '--datasets', 'demo', '--wait', '--json'], {
                    '/api/evaluation/start': [started], '/api/evaluation/status': [dict(started, status=status)]})
                self.assertEqual(code, 1)
                self.assertEqual(json.loads(stdout)['evaluation']['status'], status)
        code, stdout, _, _ = self.run_cli(['results', 'partial', '--json'], {
            '/api/runs/partial': [{'status': 'partial', 'scores': []}]})
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(stdout)['status'], 'partial')

    def test_wait_timeout_does_not_stop_background_job_or_dirty_stdout(self):
        responses = {'/api/evaluation/start': [{'status': 'running', 'log_path': 'fixture'}]}
        with patch.object(cli.time, 'monotonic', side_effect=[0, 10]):
            code, stdout, stderr, client = self.run_cli([
                'ability', 'run', '--datasets', 'demo', '--wait', '--wait-timeout', '1', '--json'], responses)
        self.assertEqual(code, 124)
        self.assertEqual(stdout, '')
        self.assertIn('后台任务仍在运行', stderr)
        self.assertEqual(len(client.calls), 1)

    def test_replaced_task_does_not_return_an_unrelated_result(self):
        code, stdout, stderr, _ = self.run_cli(['ability', 'run', '--datasets', 'demo', '--wait', '--json'], {
            '/api/evaluation/start': [{'status': 'running', 'log_path': 'original'}],
            '/api/evaluation/status': [{'status': 'completed', 'log_path': 'another', 'run_ids': ['other']}]})
        self.assertEqual(code, 2)
        self.assertEqual(stdout, '')
        self.assertIn('当前任务已变化', stderr)

    def test_reports_are_separate_from_original_ability_results(self):
        reports = [{'id': 'ai-report', 'kind': '', 'run_id': 'run-original'},
                   {'id': 'perf-report', 'kind': 'performance'}]
        code, stdout, _, client = self.run_cli(['reports', '--kind', 'analysis', '--json'], {'/api/reports': [reports]})
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(stdout), [reports[0]])
        self.assertEqual(client.calls, [('/api/reports', None)])

    def test_pelican_and_performance_payloads_and_waits_use_existing_backends(self):
        start = {'id': 'pelican-id', 'status': 'running'}
        done = dict(start, status='completed', svg='<svg/>')
        code, stdout, _, client = self.run_cli(['pelican', 'run', '--thinking-mode', 'disabled', '--max-tokens', '8192', '--wait', '--json'], {
            '/api/pelican/start': [start], '/api/reports/pelican-id': [done]})
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(stdout), done)
        self.assertEqual(client.calls[0][1], {'max_tokens': 8192, 'thinking_mode': 'disabled'})
        start = {'id': 'perf-id', 'status': 'running'}
        done = dict(start, status='completed', cells=[])
        code, stdout, _, client = self.run_cli(['performance', 'run', '--concurrency', '1,2', '--input-tokens', '1024,4096',
                                               '--requests-per-cell', '4', '--slo-ttft-ms', '1000', '--wait', '--json'], {
            '/api/performance/start': [start], '/api/performance/status': [done]})
        self.assertEqual(code, 0)
        payload = client.calls[0][1]
        self.assertEqual(payload['concurrency'], [1, 2])
        self.assertEqual(payload['input_tokens'], [1024, 4096])
        self.assertEqual(payload['slo_ttft_ms'], 1000)

    def test_transport_normalizes_errors_without_printing_response_bodies_or_credentials(self):
        api = cli.ApiClient('http://127.0.0.1:1')
        key = 'SECRET_FIXTURE_KEY'
        error = HTTPError('http://127.0.0.1:1', 400, key, {}, io.BytesIO(key.encode()))
        with patch.object(cli, 'urlopen', side_effect=error), self.assertRaises(cli.CliError) as raised:
            api.request('/api/datasets')
        self.assertIn('400', str(raised.exception))
        self.assertNotIn(key, str(raised.exception))
        with patch.object(cli, 'urlopen', side_effect=URLError(key)), self.assertRaises(cli.CliError) as raised:
            api.request('/api/datasets')
        self.assertNotIn(key, str(raised.exception))
        for server in ('http://user:SECRET@localhost:1', 'http://localhost:1?key=SECRET', 'http://localhost:bad'):
            with self.subTest(server=server), self.assertRaises(cli.CliError) as raised:
                cli.ApiClient(server)
            self.assertNotIn('SECRET', str(raised.exception))
        response = Mock()
        response.read.return_value = b'{invalid json SECRET}'
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        with patch.object(cli, 'urlopen', return_value=response), self.assertRaises(cli.CliError) as raised:
            api.request('/api/datasets')
        self.assertNotIn('SECRET', str(raised.exception))

    def test_help_is_available_without_server_imports_or_network_and_has_examples(self):
        stdout = io.StringIO()
        with patch.object(cli, 'urlopen') as network, contextlib.redirect_stdout(stdout), self.assertRaises(SystemExit) as stop:
            cli.main(['--help'])
        self.assertEqual(stop.exception.code, 0)
        network.assert_not_called()
        self.assertIn('./run_cli.sh ability run', stdout.getvalue())
        self.assertIn('results/runs', stdout.getvalue())
        for value in ('0', '-1', 'nan', 'inf'):
            with self.subTest(value=value), self.assertRaises(Exception):
                cli.positive_number(value)


if __name__ == '__main__':
    unittest.main()
