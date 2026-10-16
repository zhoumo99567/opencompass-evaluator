import json
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from performance_eval import PerformanceEvaluator, _Run, distribution, report_to_csv, summarize_cell, synthetic_prompt, validate_config

KEY = 'SECRET_PERFORMANCE_FIXTURE'


class FixtureServer(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False

    def __init__(self):
        self.mode = 'sse'
        self.seen = []
        self.received = threading.Event()
        self.finished = threading.Event()
        self.active = self.peak = 0
        self.lock = threading.Lock()
        super().__init__(('127.0.0.1', 0), FixtureHandler)
        self.thread = threading.Thread(target=self.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.shutdown()
        self.server_close()
        self.thread.join(1)

    @property
    def settings(self):
        return {'model': 'fixture-model', 'api_key': KEY, 'base_url': f'http://127.0.0.1:{self.server_port}/v1'}


class FixtureHandler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *args):
        pass

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        with self.server.lock:
            self.server.active += 1
            self.server.peak = max(self.server.peak, self.server.active)
            self.server.seen.append(payload)
        self.server.received.set()
        try:
            mode = self.server.mode
            if mode == 'http_error':
                body = json.dumps({'error': KEY}).encode()
                self.send_response(401)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if mode == 'json':
                body = json.dumps({'choices': [{'message': {'content': KEY + ' final answer'}, 'finish_reason': 'stop'}],
                                   'usage': {'prompt_tokens': 321, 'completion_tokens': 5, 'total_tokens': 326}}).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Connection', 'close')
            self.end_headers()
            self.close_connection = True

            def event(value):
                self.wfile.write(('data: ' + json.dumps(value) + '\n\n').encode())
                self.wfile.flush()

            if mode == 'stall':
                event({'choices': [{'delta': {'reasoning_content': 'starting ' + KEY}}]})
                time.sleep(2)
                event({'choices': [{'delta': {'content': 'too late'}}]})
                return
            if mode == 'invalid':
                self.wfile.write(('data: {invalid ' + KEY + '}\n\n').encode())
                self.wfile.flush()
                return
            time.sleep(.015)
            if mode != 'single_chunk':
                event({'choices': [{'delta': {'reasoning_content': 'thinking ' + KEY}}]})
                time.sleep(.015)
            if mode != 'reasoning_only':
                event({'choices': [{'delta': {'content': [{'type': 'text', 'text': 'final ' + KEY}]}}]})
            time.sleep(.015)
            event({'choices': [{'delta': {}, 'finish_reason': 'length' if mode == 'reasoning_only' else 'stop'}]})
            if mode != 'missing_usage':
                event({'choices': [], 'usage': {'prompt_tokens': 321, 'completion_tokens': 5,
                      'total_tokens': 326, 'completion_tokens_details': {'reasoning_tokens': 2}}})
            self.wfile.write(b'data: [DONE]\n\n')
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            with self.server.lock:
                self.server.active -= 1
            self.server.finished.set()


class PerformanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.server = FixtureServer()
        self.service = PerformanceEvaluator(self.tmp.name)

    def tearDown(self):
        self.service.stop()
        self.server.close()
        self.tmp.cleanup()

    def config(self, **changes):
        result = {'concurrency': [1], 'input_tokens': [128], 'requests_per_cell': 2,
                  'output_tokens': 32, 'warmup': 0, 'timeout_seconds': 2}
        result.update(changes)
        return result

    def wait_finished(self, timeout=5):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            report = self.service.status()
            if report['status'] != 'running':
                return report
            time.sleep(.01)
        self.fail('performance fixture timed out')

    def test_stream_matrix_records_real_usage_and_excludes_warmup_without_text_or_key(self):
        report = self.service.start(self.server.settings, self.config(concurrency=[1, 2], input_tokens=[128, 256], warmup=1))
        self.assertEqual(report['status'], 'running')
        with self.assertRaisesRegex(ValueError, '正在运行'):
            self.service.start(self.server.settings, self.config())
        saved = self.wait_finished()
        self.assertEqual(saved['status'], 'completed')
        self.assertEqual(saved['progress'], {'completed': 8, 'total': 8, 'current': None})
        self.assertEqual(len(saved['cells']), 4)
        self.assertEqual(len(self.server.seen), 12)
        self.assertEqual(len({request['messages'][0]['content'][:32] for request in self.server.seen}), 12)
        for request in self.server.seen:
            self.assertEqual(request['stream'], True)
            self.assertEqual(request['stream_options'], {'include_usage': True})
            self.assertEqual(request['max_tokens'], 32)
        for cell in saved['cells']:
            self.assertEqual(cell['completed'], 2)
            self.assertEqual(cell['metrics']['success_count'], 2)
            self.assertEqual(cell['actual_prompt_tokens'], {'min': 321, 'median': 321.0, 'max': 321})
            self.assertEqual(cell['actual_output_tokens'], {'min': 5, 'median': 5.0, 'max': 5})
            self.assertEqual(cell['achieved_concurrency'], cell['concurrency'])
            sample = cell['samples'][0]
            self.assertGreater(sample['ttfc_ms'], sample['ttft_ms'])
            self.assertGreater(sample['tpot_ms'], 0)
            self.assertGreater(sample['generation_tps'], 0)
            self.assertEqual(sample['usage']['reasoning_tokens'], 2)
        text = (Path(self.tmp.name) / (saved['id'] + '.json')).read_text()
        self.assertNotIn(KEY, text)
        self.assertNotIn('thinking ', text)
        self.assertEqual((Path(self.tmp.name) / (saved['id'] + '.json')).stat().st_mode & 0o777, 0o600)
        csv_data = report_to_csv(saved)
        self.assertIn('generation_tps', csv_data)
        self.assertEqual(len(csv_data.splitlines()), 9)
        self.assertNotIn(KEY, csv_data)

    def test_nonstream_response_has_e2e_and_usage_but_no_first_token_metrics(self):
        self.server.mode = 'json'
        self.service.start(self.server.settings, self.config(requests_per_cell=1, slo_ttft_ms=2000))
        saved = self.wait_finished()
        cell = saved['cells'][0]
        sample = cell['samples'][0]
        self.assertTrue(sample['success'])
        self.assertFalse(sample['streamed'])
        for metric in ('ttft_ms', 'ttfc_ms', 'tpot_ms', 'generation_tps'):
            self.assertIsNone(sample[metric])
        self.assertEqual(cell['metrics']['nonstream_count'], 1)
        self.assertEqual(cell['metrics']['slo_pass_count'], 0)
        self.assertEqual(cell['metrics']['slo_pass_rate'], 0)

    def test_missing_usage_and_single_chunk_do_not_invent_token_metrics(self):
        for mode in ('missing_usage', 'single_chunk'):
            with self.subTest(mode=mode):
                self.server.mode = mode
                self.service.start(self.server.settings, self.config(requests_per_cell=1))
                cell = self.wait_finished()['cells'][0]
                if mode == 'missing_usage':
                    self.assertEqual(cell['metrics']['missing_usage_count'], 1)
                    self.assertIsNone(cell['metrics']['output_throughput_tps'])
                    self.assertIsNone(cell['actual_prompt_tokens']['median'])
                else:
                    self.assertEqual(cell['samples'][0]['text_events'], 1)
                self.assertIsNone(cell['samples'][0]['tpot_ms'])
                self.assertIsNone(cell['samples'][0]['generation_tps'])

    def test_reasoning_only_is_flagged_separately_from_final_answers(self):
        self.server.mode = 'reasoning_only'
        self.service.start(self.server.settings, self.config(requests_per_cell=1))
        cell = self.wait_finished()['cells'][0]
        self.assertEqual(cell['metrics']['success_count'], 1)
        self.assertEqual(cell['metrics']['content_success_count'], 0)
        self.assertEqual(cell['metrics']['reasoning_only_count'], 1)
        self.assertEqual(cell['metrics']['truncated_count'], 1)
        self.assertIsNone(cell['samples'][0]['ttfc_ms'])

    def test_errors_are_safe_and_all_failed_matrix_is_failed(self):
        for mode, expected in [('http_error', 'http_error'), ('invalid', 'invalid_response')]:
            with self.subTest(mode=mode):
                self.server.mode = mode
                self.service.start(self.server.settings, self.config(requests_per_cell=1))
                saved = self.wait_finished()
                self.assertEqual(saved['status'], 'failed')
                self.assertEqual(saved['cells'][0]['samples'][0]['error_kind'], expected)
                self.assertEqual(saved['cells'][0]['metrics']['failure_count'], 1)
                self.assertNotIn(KEY, json.dumps(saved))

    def test_stop_returns_promptly_closes_active_stream_and_skips_following_cells(self):
        self.server.mode = 'stall'
        self.service.start(self.server.settings, self.config(concurrency=[1, 2], input_tokens=[128, 256], timeout_seconds=600))
        self.assertTrue(self.server.received.wait(2))
        # Wait until the response socket is registered and the initial SSE is parsed.
        time.sleep(.05)
        started = time.monotonic()
        saved = self.service.stop()
        self.assertLess(time.monotonic() - started, .5)
        self.assertEqual(saved['status'], 'cancelled')
        self.assertEqual(len(saved['cells']), 1)
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            persisted = self.service.status()
            if persisted['cells'][0]['samples']:
                break
            time.sleep(.01)
        self.assertEqual(persisted['status'], 'cancelled')
        self.assertEqual(len(persisted['cells']), 1)
        self.assertLess(len(self.server.seen), 2)
        if persisted['cells'][0]['samples']:
            self.assertEqual(persisted['cells'][0]['samples'][0]['error_kind'], 'cancelled')
        # A subsequent task can start while cancelled worker cleanup finishes.
        self.server.mode = 'json'
        self.service.start(self.server.settings, self.config(requests_per_cell=1))
        self.assertEqual(self.wait_finished()['status'], 'completed')

    def test_total_timeout_and_recovery_preserve_partial_results(self):
        self.server.mode = 'stall'
        self.service.start(self.server.settings, self.config(requests_per_cell=1, timeout_seconds=1))
        saved = self.wait_finished()
        self.assertEqual(saved['status'], 'failed')
        self.assertEqual(saved['cells'][0]['samples'][0]['error_kind'], 'timeout')
        with patch('performance_eval.threading.Thread'):
            pending = self.service.start(self.server.settings, self.config())
        recovery = PerformanceEvaluator(self.tmp.name)
        recovery.recover_interrupted()
        saved = json.loads((Path(self.tmp.name) / (pending['id'] + '.json')).read_text())
        self.assertEqual(saved['status'], 'failed')
        self.assertIn('中断', saved['error'])

    def test_live_cell_throughput_uses_current_elapsed_without_mutating_saved_results(self):
        report = {'id': 'fixture', 'status': 'running', 'config': {}, 'cells': [
            {'status': 'running', '_started': 90, 'elapsed_seconds': 1, 'samples': [
                {'success': True, 'streamed': True, 'usage': {'prompt_tokens': 100, 'completion_tokens': 20, 'total_tokens': 120},
                 'final_content_chars': 5, 'reasoning_chars': 0, 'finish_reason': 'stop',
                 'e2e_ms': 100, 'ttft_ms': 10, 'ttfc_ms': 12, 'tpot_ms': 5, 'generation_tps': 200,
                 'chunk_intervals_ms': []}]}]}
        summarize_cell(report['cells'][0])
        run = _Run(report)
        run.started = 80
        self.service.active = run
        with patch('performance_eval.time.monotonic', return_value=100):
            observed = self.service.status()
        self.assertEqual(observed['elapsed_seconds'], 20)
        self.assertEqual(observed['cells'][0]['elapsed_seconds'], 10)
        self.assertEqual(observed['cells'][0]['metrics']['output_throughput_tps'], 2)
        self.assertEqual(report['cells'][0]['elapsed_seconds'], 1)
        self.assertEqual(report['cells'][0]['metrics']['output_throughput_tps'], 20)
        self.service.active = None

    def test_cancellation_at_cell_lock_does_not_create_another_cell(self):
        report = {'id': 'fixture_cancel', 'status': 'running', 'config': validate_config(self.config()),
                  'cells': [], 'progress': {'completed': 0, 'total': 2, 'current': None},
                  'finished_at': None, 'elapsed_seconds': 0}
        run = _Run(report)
        real_lock = threading.RLock()

        class CancellationLock:
            first = True

            def __enter__(self):
                real_lock.acquire()
                # Simulate stop winning immediately after the outside-loop check.
                if self.first:
                    self.first = False
                    run.cancel.set()
                return self

            def __exit__(self, *args):
                real_lock.release()

        run.lock = CancellationLock()
        with patch.object(self.service, '_request') as request:
            self.service._run(run, self.server.settings)
        request.assert_not_called()
        self.assertEqual(report['status'], 'cancelled')
        self.assertEqual(report['cells'], [])

    def test_exception_during_warmup_finalizes_running_cell_and_keeps_report(self):
        report = {'id': 'fixture_warmup', 'status': 'running', 'config': validate_config(self.config(warmup=1)),
                  'cells': [], 'progress': {'completed': 0, 'total': 2, 'current': None},
                  'finished_at': None, 'elapsed_seconds': 0}
        run = _Run(report)
        with patch.object(self.service, '_request', side_effect=RuntimeError('fixture worker failure')):
            self.service._run(run, self.server.settings)
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(report['cells'][0]['status'], 'failed')
        self.assertNotIn('_started', report['cells'][0])
        self.assertEqual(report['cells'][0]['metrics']['success_count'], 0)
        self.assertIsNone(report['progress']['current'])

    def test_config_bounds_unique_prompt_and_metric_math(self):
        for change in ({'concurrency': [True]}, {'input_tokens': [262145]}, {'requests_per_cell': 0},
                       {'concurrency': [4], 'requests_per_cell': 2}, {'output_tokens': 8193},
                       {'timeout_seconds': 601}, {'slo_ttft_ms': 0}, {'slo_e2e_ms': float('nan')},
                       {'input_tokens': list(range(64, 101)), 'concurrency': [1]}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_config(self.config(**change))
        a, b = synthetic_prompt(1024), synthetic_prompt(1024)
        self.assertEqual(len(a), 4096)
        self.assertEqual(len(b), 4096)
        self.assertNotEqual(a[:32], b[:32])
        self.assertEqual(distribution([10, 20]), {'mean': 15.0, 'p50': 15.0, 'p95': 19.5, 'p99': 19.9})
        self.assertEqual(distribution([])['mean'], None)
        sample = {'success': True, 'streamed': True, 'usage': {'prompt_tokens': 100, 'completion_tokens': 20, 'total_tokens': 120},
                  'final_content_chars': 5, 'reasoning_chars': 0, 'finish_reason': 'stop',
                  'e2e_ms': 100, 'ttft_ms': 10, 'ttfc_ms': 12, 'tpot_ms': 5, 'generation_tps': 200,
                  'chunk_intervals_ms': [5, 10]}
        failed = dict(sample, success=False, usage={}, e2e_ms=200)
        cell = {'samples': [sample, failed], 'elapsed_seconds': 2}
        summarize_cell(cell, {'slo_ttft_ms': 20, 'slo_e2e_ms': 200})
        self.assertEqual(cell['metrics']['output_throughput_tps'], 10)
        self.assertEqual(cell['metrics']['request_throughput_rps'], .5)
        self.assertEqual(cell['metrics']['total_throughput_tps'], 60)
        self.assertEqual(cell['metrics']['error_rate'], .5)
        self.assertEqual(cell['metrics']['e2e_ms']['mean'], 100)
        self.assertEqual(cell['metrics']['slo_pass_rate'], .5)
        self.assertEqual(cell['metrics']['goodput_rps'], .5)


if __name__ == '__main__':
    unittest.main()
