import unittest
from unittest.mock import Mock, patch

import web_viewer as viewer


class EnvironmentRequestTests(unittest.TestCase):
    def headers(self, **extra):
        return {'Host': '127.0.0.1:8766', 'Content-Type': 'application/json',
                'Content-Length': '25', **extra}

    def test_environment_actions_reject_foreign_origins_and_nonlocal_clients(self):
        allowed = viewer._local_environment_request
        self.assertTrue(allowed(('127.0.0.1', 1), self.headers(), 8766))
        self.assertTrue(allowed(('127.0.0.1', 1), self.headers(Origin='http://127.0.0.1:8766'), 8766))
        for extra in ({'Origin': 'https://example.com'},
                      {'Origin': 'http://127.0.0.1:8765'},
                      {'Origin': 'null'}, {'Content-Type': 'text/plain'},
                      {'Host': 'example.com:8766'}, {'Content-Length': '9000'},
                      {'Content-Length': '-1'}, {'Content-Length': 'invalid'}):
            with self.subTest(headers=extra):
                self.assertFalse(allowed(('127.0.0.1', 1), self.headers(**extra), 8766))
        self.assertFalse(allowed(('192.168.6.2', 1), self.headers(), 8766))

    def test_unknown_action_never_reaches_the_environment_manager(self):
        with patch.object(viewer, 'ENVIRONMENT') as manager:
            with self.assertRaisesRegex(ValueError, '未知'):
                viewer._start_environment('arbitrary-command')
            manager.start.assert_not_called()

    def test_installation_and_evaluation_cannot_start_together(self):
        factory = Mock()
        with patch.object(viewer, 'ENVIRONMENT') as manager:
            manager.status.return_value = {'busy': True}
            with self.assertRaisesRegex(ValueError, '环境任务'):
                viewer._start_activity(factory)
            factory.assert_not_called()
            manager.status.return_value = {'busy': False}
            factory.return_value = {'status': 'running'}
            self.assertEqual(viewer._start_activity(factory), {'status': 'running'})
        with patch.object(viewer, '_eval_public_state', return_value={'status': 'running'}), \
                patch.object(viewer, 'ENVIRONMENT') as manager:
            with self.assertRaisesRegex(ValueError, '评测正在运行'):
                viewer._start_environment('install_code')
            manager.start.assert_not_called()


if __name__ == '__main__':
    unittest.main()
