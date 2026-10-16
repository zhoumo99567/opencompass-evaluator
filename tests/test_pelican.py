import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from pelican_eval import MAX_TOKENS, PelicanEvaluator, PROMPT, http_error_message, sanitize_svg

SVG = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 300"><circle cx="100" cy="200" r="50" fill="none" stroke="black"/><circle cx="300" cy="200" r="50" fill="none" stroke="black"/><text x="80" y="80">测试绘图</text></svg>'
SETTINGS = dict(model='fixture-model', api_key='SECRET_TEST_TOKEN', base_url='http://127.0.0.1:1/v1')


class PelicanTests(unittest.TestCase):
    def test_generation_budget_is_validated_and_persisted(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = PelicanEvaluator(tmp)
            for value in (0, True, '32768', 262145, None):
                with self.subTest(value=value), self.assertRaisesRegex(ValueError, '生成额度'):
                    service.start(SETTINGS, max_tokens=value)
            with patch('pelican_eval.threading.Thread'):
                report = service.start(SETTINGS, max_tokens=262144)
            self.assertEqual(report['max_tokens'], 262144)
            saved = json.loads((Path(tmp) / (report['id'] + '.json')).read_text())
            self.assertEqual(saved['max_tokens'], 262144)

    def test_extracts_fenced_svg_and_preserves_drawing(self):
        svg, removed = sanitize_svg('```svg\n' + SVG + '\n```')
        self.assertIn('测试绘图', svg)
        self.assertIn('viewBox="0 0 400 300"', svg)
        self.assertEqual(removed, 0)

    def test_removes_scripts_external_references_and_event_handlers(self):
        source = '''<svg xmlns="http://www.w3.org/2000/svg" onload="alert(1)">
        <script>alert(1)</script><foreignObject><div>unsafe</div></foreignObject>
        <image href="https://example.invalid/a"/>
        <use href="https://example.invalid/a#p"/>
        <path d="M0 0L1 1" style="stroke:red;fill:url(https://example.invalid/a);behavior:expression(a)"/>
        <use href="#safe"/><path fill="url(#paint)"/>
        </svg>'''
        svg, removed = sanitize_svg(source)
        for unsafe in ('onload', 'script', 'foreignObject', 'https:', 'expression'):
            self.assertNotIn(unsafe, svg)
        self.assertIn('stroke:red;', svg)
        self.assertIn('href="#safe"', svg)
        self.assertGreater(removed, 0)

    def test_css_animation_classes_and_reduced_motion_are_preserved(self):
        source = '''<svg xmlns="http://www.w3.org/2000/svg">
        <style>.spin{transform-box:fill-box;transform-origin:center;animation:spin 1s linear infinite}
        @keyframes spin{from{transform:rotate(0deg)}to{transform:rotate(360deg)}}
        @media (prefers-reduced-motion:reduce){*{animation:none!important}}</style>
        <g class="spin"><path d="M0 0L20 20"/></g>
        <g style="animation:spin 2s linear infinite;transform-origin:center"/>
        </svg>'''
        svg, removed = sanitize_svg(source)
        self.assertIn('@keyframes spin', svg)
        self.assertIn('class="spin"', svg)
        self.assertIn('prefers-reduced-motion', svg)
        self.assertIn('animation:spin 2s', svg)
        self.assertEqual(removed, 0)
        self.assertEqual(sanitize_svg(svg)[0], svg)

    def test_stylesheets_cannot_import_or_fetch_external_resources(self):
        source = r'''<svg xmlns="http://www.w3.org/2000/svg"><style>
        @import url(https://example.invalid/a.css);
        @font-face{font-family:bad;src:url(https://example.invalid/f)}
        .spin{fill:url(https://example.invalid/a);stroke:red;animation:spin 1s infinite}
        .escape{fill:\75rl(https://example.invalid/a)}
        @keyframes spin{to{transform:rotate(360deg);fill:url(https://example.invalid/a)}}
        </style><script>alert(1)</script><g class="spin" onload="alert(1)"/></svg>'''
        svg, removed = sanitize_svg(source)
        for unsafe in ('https:', '@import', '@font-face', 'onload', '<script'):
            self.assertNotIn(unsafe, svg)
        self.assertIn('@keyframes spin', svg)
        self.assertIn('stroke:red', svg)
        self.assertGreater(removed, 0)

    def test_invalid_and_entity_svg_rejected(self):
        for source in ('no drawing', '<svg><path></svg>', '<!DOCTYPE svg [<!ENTITY a "x">]><svg>&a;</svg>'):
            with self.subTest(source=source), self.assertRaises(ValueError):
                sanitize_svg(source)

    def test_request_persistence_and_failure_keep_raw_output(self):
        for content, status in ((SVG, 'completed'), ('not an SVG SECRET_TEST_TOKEN', 'failed')):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as tmp:
                service = PelicanEvaluator(tmp)
                response = Mock()
                response.read.return_value = json.dumps({'choices': [{'message': {'content': content}, 'finish_reason': 'stop'}], 'usage': {'completion_tokens': 12}}).encode()
                response.status = 200
                response.getheader.return_value = 'application/json'
                connection = Mock(sock=None)
                connection.getresponse.return_value = response
                with patch('pelican_eval.threading.Thread') as thread:
                    report = service.start(SETTINGS)
                    self.assertEqual(report['status'], 'running')
                    with self.assertRaisesRegex(ValueError, '正在生成'):
                        service.start(SETTINGS)
                    self.assertTrue(thread.return_value.start.called)
                with patch('pelican_eval.http.client.HTTPConnection', return_value=connection):
                    service._run(report, SETTINGS)
                    self.assertEqual(connection.request.call_args.args[:2], ('POST', '/v1/chat/completions'))
                    payload = json.loads(connection.request.call_args.kwargs['body'])
                    self.assertTrue(payload['stream'])
                    self.assertEqual(payload['stream_options'], {'include_usage': True})
                    self.assertNotIn('chat_template_kwargs', payload)
                    self.assertEqual(payload['messages'], [{'role': 'user', 'content': PROMPT}])
                    self.assertEqual(payload['max_tokens'], MAX_TOKENS)
                saved = json.loads((Path(tmp)/(report['id']+'.json')).read_text())
                self.assertEqual(saved['status'], status)
                self.assertEqual(saved['usage']['completion_tokens'], 12)
                self.assertNotIn(SETTINGS['api_key'], json.dumps(saved))
                self.assertIsNone(service.active_id)

    def test_reasoning_only_length_preserves_diagnostics_without_claiming_success(self):
        saved = self.run_response({
            'content': None, 'reasoning': SVG,
        }, finish_reason='length', usage={
            'completion_tokens': 32768,
            'completion_tokens_details': {'reasoning_tokens': 32768},
        })
        self.assertEqual(saved['status'], 'failed')
        self.assertEqual(saved['finish_reason'], 'length')
        self.assertEqual(saved['usage']['reasoning_tokens'], 32768)
        self.assertEqual(saved['reasoning_chars'], len(SVG))
        self.assertIn('生成额度', saved['error'])
        self.assertEqual(saved['content'], '')
        self.assertEqual(saved['svg'], '')

    def test_content_blocks_are_joined_without_reasoning_blocks(self):
        saved = self.run_response({'content': [
            {'type': 'reasoning', 'text': 'not final'},
            {'type': 'text', 'text': SVG[:50]},
            {'type': 'text', 'text': SVG[50:]},
        ]})
        self.assertEqual(saved['status'], 'completed')
        self.assertEqual(saved['content'], SVG)

    def test_empty_refusal_and_reasoning_only_responses_have_specific_errors(self):
        for message, expected in [
            ({'content': None}, '空文本'),
            ({'content': '', 'refusal': 'no'}, '拒绝'),
            ({'content': None, 'reasoning_content': 'thinking'}, '思考内容'),
        ]:
            with self.subTest(message=message):
                saved = self.run_response(message)
                self.assertEqual(saved['status'], 'failed')
                self.assertIn(expected, saved['error'])

    def run_response(self, message, finish_reason='stop', usage=None):
        with tempfile.TemporaryDirectory() as tmp:
            service = PelicanEvaluator(tmp)
            with patch('pelican_eval.threading.Thread'):
                report = service.start(SETTINGS)
            response = Mock()
            response.read.return_value = json.dumps({
                'choices': [{'message': message, 'finish_reason': finish_reason}],
                'usage': usage or {},
            }).encode()
            response.status = 200
            response.getheader.return_value = 'application/json'
            connection = Mock(sock=None)
            connection.getresponse.return_value = response
            with patch('pelican_eval.http.client.HTTPConnection', return_value=connection):
                service._run(report, SETTINGS)
            self.assertIsNone(service.active_id)
            return json.loads((Path(tmp) / (report['id'] + '.json')).read_text())

    def test_timeout_is_saved_and_unlocks_next_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = PelicanEvaluator(tmp)
            with patch('pelican_eval.threading.Thread'):
                report = service.start(SETTINGS)
            connection = Mock(sock=None)
            connection.request.side_effect = TimeoutError('SECRET_TEST_TOKEN')
            with patch('pelican_eval.http.client.HTTPConnection', return_value=connection):
                service._run(report, SETTINGS)
            saved = json.loads((Path(tmp)/(report['id']+'.json')).read_text())
            self.assertEqual(saved['status'], 'failed')
            self.assertNotIn('SECRET_TEST_TOKEN', saved['error'])
            self.assertIsNone(service.active_id)

    def test_restart_marks_pending_report_interrupted(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = PelicanEvaluator(tmp)
            with patch('pelican_eval.threading.Thread'):
                report = service.start(SETTINGS)
            PelicanEvaluator(tmp).recover_interrupted()
            saved = json.loads((Path(tmp)/(report['id']+'.json')).read_text())
            self.assertEqual(saved['status'], 'failed')
            self.assertIn('中断', saved['error'])


ANIMATED_SVG = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 300"><style>.wheel{animation:spin 1s linear infinite;transform-origin:center}@keyframes spin{to{transform:rotate(360deg)}}</style><g class="wheel"><circle cx="100" cy="200" r="50"/></g><desc>SECRET_TEST_TOKEN</desc></svg>'


class PelicanFixtureServer(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False

    def __init__(self):
        self.mode = 'complete'
        self.received = threading.Event()
        self.release = threading.Event()
        self.payloads = []
        super().__init__(('127.0.0.1', 0), PelicanFixtureHandler)
        self.thread = threading.Thread(target=self.serve_forever, daemon=True)
        self.thread.start()

    @property
    def settings(self):
        return dict(model='fixture-model', api_key=SETTINGS['api_key'], base_url=f'http://127.0.0.1:{self.server_port}/v1')

    def close(self):
        self.release.set()
        self.shutdown()
        self.server_close()
        self.thread.join(1)


class PelicanFixtureHandler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *args):
        pass

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.server.payloads.append(payload)
        self.server.received.set()
        mode = self.server.mode
        try:
            if mode.startswith('http'):
                code = int(mode[4:])
                body = json.dumps({'error': {'message': SETTINGS['api_key'] + ' PRIVATE_GATEWAY_BODY'}}).encode()
                self.send_response(code)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Connection', 'close')
                self.end_headers()
                self.close_connection = True
                self.wfile.write(body)
                return
            if mode == 'json':
                body = json.dumps({'choices': [{'message': {'content': ANIMATED_SVG}, 'finish_reason': 'stop'}],
                                   'usage': {'completion_tokens': 10}}).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Connection', 'close')
                self.end_headers()
                self.close_connection = True
                self.wfile.write(body)
                return
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Connection', 'close')
            self.end_headers()
            self.close_connection = True

            def event(data):
                self.wfile.write(('data: ' + json.dumps(data) + '\n\n').encode())
                self.wfile.flush()

            event({'choices': [{'delta': {'reasoning_content': 'thinking SECRET_TEST_TOKEN'}}]})
            if mode != 'reasoning_only':
                part = ANIMATED_SVG if mode == 'cut_complete' else ANIMATED_SVG[:100]
                event({'choices': [{'delta': {'content': [{'type': 'text', 'text': part}]}}]})
            if mode == 'paused':
                self.server.release.wait(3)
            if mode in ('cut', 'cut_complete'):
                return
            if mode == 'malformed':
                self.wfile.write(b'data: {invalid}\n\n')
                self.wfile.flush()
                return
            if mode == 'drip':
                for _ in range(50):
                    time.sleep(.02)
                    event({'choices': [{'delta': {'reasoning': 'still thinking'}}]})
            if mode != 'reasoning_only':
                event({'choices': [{'delta': {'content': ANIMATED_SVG[100:]}}]})
            event({'choices': [{'delta': {}, 'finish_reason': 'length' if mode == 'reasoning_only' else 'stop'}]})
            event({'choices': [], 'usage': {'completion_tokens': 20, 'prompt_tokens': 10,
                   'total_tokens': 30, 'completion_tokens_details': {'reasoning_tokens': 12}}})
            self.wfile.write(b'data: [DONE]\n\n')
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass


class PelicanStreamingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.server = PelicanFixtureServer()
        self.service = PelicanEvaluator(self.tmp.name)

    def tearDown(self):
        self.service.stop()
        self.server.close()
        deadline = time.monotonic() + 2
        while self.service.active_id and time.monotonic() < deadline:
            time.sleep(.01)
        self.tmp.cleanup()

    def read(self, report_id):
        return json.loads((Path(self.tmp.name) / (report_id + '.json')).read_text())

    def wait_finished(self, report_id):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            report = self.read(report_id)
            if report['status'] != 'running' and self.service.active_id is None:
                return report
            time.sleep(.01)
        self.fail('pelican fixture timeout')

    def test_live_partial_is_persisted_then_completed_svg_retains_animation_and_usage(self):
        self.server.mode = 'paused'
        report = self.service.start(self.server.settings, max_tokens=131072)
        self.assertTrue(self.server.received.wait(2))
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            partial = self.read(report['id'])
            if partial['content'] and partial['reasoning_chars'] and partial['elapsed_seconds'] >= 1:
                break
            time.sleep(.02)
        self.assertEqual(partial['status'], 'running')
        self.assertEqual(partial['phase'], 'writing')
        self.assertTrue(partial['streamed'])
        self.assertTrue(partial['content'])
        self.assertEqual(partial['svg'], '')
        self.assertGreater(partial['elapsed_seconds'], .9)
        self.assertIsNotNone(partial['first_response_seconds'])
        self.assertNotIn(SETTINGS['api_key'], json.dumps(partial))
        self.server.release.set()
        saved = self.wait_finished(report['id'])
        self.assertEqual(saved['status'], 'completed')
        self.assertTrue(saved['animated'])
        self.assertEqual(saved['usage']['completion_tokens'], 20)
        self.assertEqual(saved['usage']['reasoning_tokens'], 12)
        self.assertEqual(saved['finish_reason'], 'stop')
        self.assertNotIn(SETTINGS['api_key'], json.dumps(saved))
        self.assertNotIn('thinking SECRET', json.dumps(saved))
        payload = self.server.payloads[-1]
        self.assertTrue(payload['stream'])
        self.assertEqual(payload['stream_options'], {'include_usage': True})
        self.assertNotIn('chat_template_kwargs', payload)

    def test_cut_malformed_and_unfinished_complete_svg_keep_partial_but_fail(self):
        for mode in ('cut', 'cut_complete', 'malformed'):
            with self.subTest(mode=mode):
                self.server.mode = mode
                report = self.service.start(self.server.settings)
                saved = self.wait_finished(report['id'])
                self.assertEqual(saved['status'], 'failed')
                self.assertEqual(saved['error_kind'], 'invalid_response')
                self.assertTrue(saved['content'].startswith('<svg'))
                self.assertGreater(saved['reasoning_chars'], 0)
                self.assertEqual(saved['svg'], '')
                self.assertNotIn(SETTINGS['api_key'], json.dumps(saved))

    def test_stream_reasoning_only_keeps_diagnostics_and_no_false_svg(self):
        self.server.mode = 'reasoning_only'
        report = self.service.start(self.server.settings)
        saved = self.wait_finished(report['id'])
        self.assertEqual(saved['status'], 'failed')
        self.assertEqual(saved['error_kind'], 'budget_exhausted')
        self.assertEqual(saved['content'], '')
        self.assertGreater(saved['reasoning_chars'], 0)
        self.assertEqual(saved['usage']['reasoning_tokens'], 12)
        self.assertEqual(saved['finish_reason'], 'length')

    def test_http_errors_identify_gateway_credentials_parameters_and_load_without_body(self):
        for code, fragment in ((504, '网关'), (401, '认证'), (403, '认证'), (400, '扩展参数'), (429, '限流'), (502, '网关')):
            with self.subTest(code=code):
                self.server.mode = 'http' + str(code)
                report = self.service.start(self.server.settings, thinking_mode='disabled')
                saved = self.wait_finished(report['id'])
                self.assertEqual(saved['http_status'], code)
                self.assertEqual(saved['error_kind'], 'http_error')
                self.assertIn(fragment, saved['error'])
                self.assertNotIn('PRIVATE_GATEWAY_BODY', json.dumps(saved))
                self.assertNotIn(SETTINGS['api_key'], json.dumps(saved))
        self.assertNotIn('密钥', http_error_message(504))
        self.assertIn('总时限', http_error_message(504))

    def test_cancellation_returns_promptly_is_idempotent_and_slot_is_worker_owned(self):
        self.server.mode = 'paused'
        report = self.service.start(self.server.settings)
        self.assertTrue(self.server.received.wait(2))
        time.sleep(.03)
        started = time.monotonic()
        stopped = self.service.stop()
        self.assertLess(time.monotonic() - started, .5)
        self.assertEqual(stopped['status'], 'cancelled')
        self.assertEqual(self.service.stop()['status'], 'cancelled')
        saved = self.wait_finished(report['id'])
        self.assertEqual(saved['status'], 'cancelled')
        self.assertTrue(saved['content'])
        # Prevent worker exit to deterministically test the slot guard.
        with patch('pelican_eval.threading.Thread'):
            pending = self.service.start(self.server.settings)
        self.service.stop()
        self.assertEqual(self.service.active_id, pending['id'])
        with self.assertRaisesRegex(ValueError, '正在生成或停止'):
            self.service.start(self.server.settings)
        self.service._run(pending, self.server.settings)
        self.assertIsNone(self.service.active_id)

    def test_overall_deadline_cuts_an_active_stream_despite_frequent_events(self):
        self.server.mode = 'drip'
        with patch('pelican_eval.REQUEST_TIMEOUT_SECONDS', .15):
            started = time.monotonic()
            report = self.service.start(self.server.settings)
            saved = self.wait_finished(report['id'])
        self.assertLess(time.monotonic() - started, .6)
        self.assertEqual(saved['status'], 'failed')
        self.assertEqual(saved['error_kind'], 'timeout')
        self.assertTrue(saved['content'])
        self.assertGreater(saved['reasoning_chars'], 0)
        self.assertGreater(saved['received_events'], 2)

    def test_http_error_cannot_overwrite_a_concurrent_stop(self):
        with patch('pelican_eval.threading.Thread'):
            report = self.service.start(self.server.settings)
        response = Mock(status=504)
        response.getheader.return_value = 'application/json'
        connection = Mock(sock=None)

        def headers_arrive():
            self.service.stop()
            return response

        connection.getresponse.side_effect = headers_arrive
        with patch('pelican_eval.http.client.HTTPConnection', return_value=connection):
            self.service._run(report, self.server.settings)
        saved = self.read(report['id'])
        self.assertEqual(saved['status'], 'cancelled')
        self.assertEqual(saved['error_kind'], 'cancelled')
        self.assertIsNone(saved['error'])
        self.assertIsNone(self.service.active_id)

    def test_disabled_thinking_payload_and_json_fallback(self):
        self.server.mode = 'json'
        report = self.service.start(self.server.settings, max_tokens=8192, thinking_mode='disabled')
        saved = self.wait_finished(report['id'])
        self.assertEqual(saved['status'], 'completed')
        self.assertFalse(saved['streamed'])
        self.assertEqual(saved['thinking_mode'], 'disabled')
        payload = self.server.payloads[-1]
        self.assertEqual(payload['chat_template_kwargs'], {'enable_thinking': False})
        self.assertEqual(payload['max_tokens'], 8192)
        for value in (None, True, 'unknown'):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, '思考模式'):
                self.service.start(self.server.settings, thinking_mode=value)


if __name__ == '__main__':
    unittest.main()
