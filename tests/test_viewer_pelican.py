import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import web_viewer


class PelicanReportViewTests(unittest.TestCase):
    def load_report(self, report):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / 'fixture.json'
            path.write_text(json.dumps(report), encoding='utf-8')
            with patch.object(web_viewer, 'REPORTS', root):
                result = web_viewer.report_detail('fixture')
            self.assertEqual(json.loads(path.read_text()), report)
            return result

    def test_legacy_504_explains_gateway_timeout_without_changing_saved_history(self):
        report = dict(id='fixture', kind='pelican', status='failed',
                      error='模型接口返回 HTTP 504，请检查模型设置',
                      max_tokens=131072, elapsed_seconds=180.01)
        viewed = self.load_report(report)
        self.assertIn('504', viewed['error'])
        self.assertIn('超时', viewed['error'])
        self.assertNotIn('请检查模型设置', viewed['error'])
        self.assertEqual(viewed['http_status'], 504)
        self.assertEqual(viewed['elapsed_seconds'], 180.01)
        self.assertEqual(viewed['max_tokens'], 131072)

    def test_running_200_with_partial_text_does_not_become_an_http_failure(self):
        report = dict(id='fixture', kind='pelican', status='running',
                      http_status=200, content='<svg', error=None)
        self.assertEqual(self.load_report(report), report)

    def test_completed_200_retains_successful_animated_preview(self):
        source = '<svg xmlns="http://www.w3.org/2000/svg"><style>.wheel{animation:spin 1s infinite}@keyframes spin{to{transform:rotate(360deg)}}</style><circle class="wheel" r="10"/></svg>'
        viewed = self.load_report(dict(id='fixture', kind='pelican', status='completed',
                                      http_status=200, content=source, svg=source, error=None))
        self.assertIsNone(viewed['error'])
        self.assertEqual(viewed['status'], 'completed')
        self.assertTrue(viewed['animated'])


if __name__ == '__main__':
    unittest.main()
