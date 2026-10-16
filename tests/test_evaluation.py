import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from mmengine.config import Config

import web_viewer as viewer
from evaluation_config import build_evaluation_config, validate_evaluation_options

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = dict(model='test-model', api_key='test-key', base_url='http://127.0.0.1:1/v1')


class EvaluationTests(unittest.TestCase):
    def test_summary_preserves_low_percentage_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / 'summary').mkdir()
            (run / 'summary' / 'scores.csv').write_text(
                'dataset,metric,model\ngsm8k,accuracy,0.5\nmath,accuracy,1\ncmmlu,accuracy,87.5%\n')
            self.assertEqual([x['score'] for x in viewer.parse_summary(run)], [0.5, 1.0, 87.5])

    def test_all_supported_configs_survive_serialization_without_collisions(self):
        selected = [item['id'] for item in viewer.dataset_catalog() if item['available']]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'eval.py'
            manifest = build_evaluation_config(ROOT, selected, SETTINGS, path)
            config = Config.fromfile(str(path))
            names = [item['abbr'] for item in config.datasets]
            self.assertEqual([item['id'] for item in manifest], selected)
            self.assertEqual(len(names), len(set(names)))
            self.assertEqual(set(names), {n for item in manifest for n in item['abbrs']})
            for dataset_id in selected:
                self.assertTrue(any(name.startswith(dataset_id + '__') for name in names))
            self.assertEqual(config.models[0]['openai_api_base'], SETTINGS['base_url'] + '/chat/completions')

    def test_launch_uses_generated_config_and_every_selection(self):
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(viewer, 'OUTPUTS', Path(tmp)), \
             patch.object(viewer, 'load_settings', return_value=SETTINGS), \
             patch.object(viewer, 'dataset_catalog', return_value=[
                 dict(id='demo_gsm8k_chat_gen', available=True),
                 dict(id='demo_math_chat_gen', available=True)]), \
             patch.dict(viewer.EVAL_STATE, {'process': None}, clear=True), \
             patch.object(viewer.subprocess, 'Popen') as popen:
            popen.return_value = Mock(pid=123, poll=Mock(return_value=None))
            selected = ['demo_gsm8k_chat_gen', 'demo_math_chat_gen']
            state = viewer.start_evaluation(selected)
            args = popen.call_args.args[0]
            self.assertNotIn('--datasets', args)
            self.assertNotEqual(args[1], str(ROOT / 'lan_eval.py'))
            names = [d['abbr'] for d in Config.fromfile(args[1]).datasets]
            self.assertEqual(len(names), 2)
            for dataset_id in selected:
                self.assertTrue(any(name.startswith(dataset_id + '__') for name in names))
            self.assertEqual(state['datasets'], selected)
            self.assertEqual(state['dataset_count'], 2)
            with self.assertRaisesRegex(ValueError, '正在运行'):
                viewer.start_evaluation(selected)

    def test_explicit_budgets_override_models_and_short_dataset_defaults(self):
        selected = ['longbench_passage_retrieval_zh_gen',
                    'longbench_multifieldqa_zh_gen', 'math_gen_265cce', 'gsm8k_gen']
        options = dict(max_seq_len=262144, max_out_len=32768)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'eval.py'
            manifest = build_evaluation_config(ROOT, selected, SETTINGS, path, options)
            config = Config.fromfile(str(path))
            self.assertEqual([item['id'] for item in manifest], selected)
            for model in config.models:
                self.assertEqual(model.max_seq_len, options['max_seq_len'])
                self.assertEqual(model.max_out_len, options['max_out_len'])
                self.assertEqual(model.mode, 'none')
                self.assertEqual(model.path, SETTINGS['model'])
            self.assertEqual(len(config.datasets), len(selected))
            for dataset in config.datasets:
                self.assertEqual(dataset.infer_cfg.inferencer.max_out_len, options['max_out_len'])
                inferencer_type = dataset.infer_cfg.inferencer.type
                type_name = inferencer_type if isinstance(inferencer_type, str) else inferencer_type.__name__
                self.assertEqual(type_name.rsplit('.', 1)[-1], 'GenInferencer')
                self.assertTrue(dataset.infer_cfg.prompt_template)
                self.assertTrue(dataset.eval_cfg.evaluator)
            self.assertEqual(options, dict(max_seq_len=262144, max_out_len=32768))

    def test_omitted_budgets_preserve_upstream_and_base_model_config(self):
        selected = ['longbench_passage_retrieval_zh_gen', 'math_gen_265cce']
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'eval.py'
            build_evaluation_config(ROOT, selected, SETTINGS, path)
            config = Config.fromfile(str(path))
            self.assertEqual([d.infer_cfg.inferencer.max_out_len for d in config.datasets], [32, 512])
            base = Config.fromfile(str(ROOT / 'lan_eval.py'))
            for key in ('max_seq_len', 'max_out_len', 'mode'):
                self.assertEqual(config.models[0].get(key), base.models[0].get(key))

    def test_invalid_budgets_rejected_before_creating_config(self):
        good = dict(max_seq_len=262144, max_out_len=32768)
        invalid = [[], {}, {'max_seq_len': 128},
                   dict(good, max_seq_len=63), dict(good, max_seq_len=262145),
                   dict(good, max_seq_len=True), dict(good, max_seq_len='262144'),
                   dict(good, max_out_len=63), dict(good, max_out_len=65537),
                   dict(good, max_out_len=128.0), dict(good, max_out_len=False),
                   dict(max_seq_len=32768, max_out_len=32768),
                   dict(max_seq_len=128, max_out_len=256)]
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / 'not-created' / 'eval.py'
            for options in invalid:
                with self.subTest(options=options), self.assertRaises(ValueError):
                    build_evaluation_config(ROOT, ['gsm8k_gen'], SETTINGS, destination, options)
                self.assertFalse(destination.parent.exists())
        self.assertIsNone(validate_evaluation_options(None))
        self.assertEqual(validate_evaluation_options(dict(max_seq_len=128, max_out_len=64)),
                         dict(max_seq_len=128, max_out_len=64))
        self.assertEqual(validate_evaluation_options(dict(max_seq_len=262144, max_out_len=65536)),
                         dict(max_seq_len=262144, max_out_len=65536))

    def test_invalid_and_ppl_selection_rejected_before_launch(self):
        catalog = [dict(id='demo_gsm8k_chat_gen', available=True),
                   dict(id='demo_cmmlu_base_ppl', available=False, unsupported_reason='PPL unsupported')]
        with patch.object(viewer, 'dataset_catalog', return_value=catalog), \
                patch.object(viewer.subprocess, 'Popen') as popen:
            for selected in ([], ['unknown'], 'demo_gsm8k_chat_gen', ['demo_cmmlu_base_ppl']):
                with self.subTest(selected=selected), self.assertRaises(ValueError):
                    viewer.start_evaluation(selected)
            popen.assert_not_called()

    def test_partial_results_not_marked_completed(self):
        with tempfile.TemporaryDirectory() as tmp:
            job = Path(tmp)
            run = job / '20261011_120000'
            results = run / 'results' / 'test-model'
            results.mkdir(parents=True)
            (job / 'evaluation_manifest.json').write_text(json.dumps([
                {'id': 'a', 'abbrs': ['a']}, {'id': 'b', 'abbrs': ['b']}]))
            (results / 'a.json').write_text('{"accuracy": 50}')
            summary = viewer.run_summary(run)
            self.assertEqual(summary['status'], 'partial')
            self.assertEqual(summary['coverage']['completed'], 1)
            with patch.dict(viewer.EVAL_STATE, {
                'process': Mock(poll=Mock(return_value=0), returncode=0),
                'status': 'running', 'work_dir': str(job)
            }, clear=True):
                self.assertEqual(viewer._eval_public_state()['status'], 'failed')
            (results / 'b.json').write_text('{"accuracy": 75}')
            self.assertEqual(viewer.run_summary(run)['status'], 'completed')

    def test_status_exposes_only_current_job_run_ids_for_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            job = Path(tmp) / 'job'
            job.mkdir()
            for name in ('20261015_120000', '20261015_120010', 'configs'):
                (job / name).mkdir()
            # Files and nested timestamps are not OpenCompass run directories.
            (job / '20261015_120020').write_text('fixture')
            (job / 'configs' / '20261015_120030').mkdir()
            with patch.dict(viewer.EVAL_STATE, {
                'status': 'running', 'process': Mock(poll=Mock(return_value=None)),
                'work_dir': str(job),
            }, clear=True):
                public = viewer._eval_public_state()
                self.assertEqual(public['run_ids'], ['20261015_120010', '20261015_120000'])
                self.assertNotIn('work_dir', public)
                self.assertNotIn('process', public)
            with patch.dict(viewer.EVAL_STATE, {'status': 'idle'}, clear=True):
                self.assertEqual(viewer._eval_public_state()['run_ids'], [])


if __name__ == '__main__':
    unittest.main()
