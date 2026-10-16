import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mmengine.config import Config

from benchmark_catalog import (BENCHMARK_CATEGORIES, BENCHMARK_PRESETS,
                               DATASET_CATALOG, dataset_preparation,
                               describe_benchmark, unavailable_reason)
from evaluation_config import build_evaluation_config


ROOT = Path(__file__).resolve().parents[1]


class BenchmarkCatalogTests(unittest.TestCase):
    def test_current_ids_and_all_declared_installed_configs_load(self):
        old_ids = ['demo_gsm8k_chat_gen', 'demo_math_chat_gen',
                   'demo_cmmlu_chat_gen', 'demo_gsm8k_base_gen',
                   'demo_math_base_gen', 'demo_cmmlu_base_ppl', 'humaneval_cn_gen']
        self.assertEqual([item['id'] for item in DATASET_CATALOG[:7]], old_ids)
        config_root = next((ROOT / '.venv/lib').glob(
            'python*/site-packages/opencompass/configs/datasets'))
        for item in DATASET_CATALOG:
            with self.subTest(dataset=item['id']):
                matches = list(config_root.rglob(item['id'] + '.py'))
                if item['integration'] == 'external':
                    self.assertEqual(matches, [])
                    self.assertTrue(unavailable_reason(item, ROOT))
                    continue
                self.assertEqual(len(matches), 1)
                config = Config.fromfile(str(matches[0]))
                datasets = [d for key in config.keys()
                            if key.endswith('_datasets') for d in config[key]]
                self.assertTrue(datasets)
                for dataset in datasets:
                    prefixed = item['id'] + '__' + dataset['abbr']
                    for name in (prefixed, prefixed.replace('_', ' ')):
                        explanation = describe_benchmark(name)
                        self.assertEqual(explanation['family'], item['family'])
                        self.assertEqual(explanation['dataset'], name)
                        self.assertEqual(explanation['category_id'], item['category_id'])
                        for field in ('purpose', 'task_example', 'suitable_uses'):
                            self.assertTrue(explanation[field])

    def test_all_picker_entries_include_visible_use_and_variant_explanations(self):
        entries = {item['id']: item for item in DATASET_CATALOG}
        for item in DATASET_CATALOG:
            with self.subTest(dataset=item['id']):
                self.assertTrue(item['purpose'])
                self.assertTrue(item['task_example'])
                self.assertTrue(item['suitable_uses'])
        self.assertIn('少量', entries['demo_gsm8k_chat_gen']['purpose'])
        self.assertIn('Base', entries['demo_math_base_gen']['purpose'])
        self.assertIn('当前聊天接口', entries['demo_cmmlu_base_ppl']['purpose'])
        self.assertIn('中文', entries['humaneval_cn_gen']['purpose'])
        self.assertIn('测试', entries['mbpp_gen']['purpose'])

    def test_taxonomy_and_presets_keep_stable_ids_and_real_prerequisites(self):
        category_ids = [item['id'] for item in BENCHMARK_CATEGORIES]
        self.assertEqual(len(category_ids), len(set(category_ids)))
        self.assertEqual([item['order'] for item in BENCHMARK_CATEGORIES],
                         sorted(item['order'] for item in BENCHMARK_CATEGORIES))
        categories = {item['id']: item for item in BENCHMARK_CATEGORIES}
        entries = {item['id']: item for item in DATASET_CATALOG}
        self.assertEqual(len(entries), len(DATASET_CATALOG))
        self.assertEqual([item['id'] for item in DATASET_CATALOG[7:17]],
                         ['gsm8k_gen', 'math_gen_265cce', 'cmmlu_gen', 'mmlu_gen',
                          'bbh_gen', 'IFEval_gen', 'longbench_multifieldqa_zh_gen',
                          'longbench_passage_retrieval_zh_gen', 'truthfulqa_gen', 'mbpp_gen'])
        for entry in DATASET_CATALOG:
            self.assertIn(entry['category_id'], categories)
            self.assertEqual(entry['category'], categories[entry['category_id']]['label'])
            self.assertIn(entry['scope'], {'demo', 'full'})
            self.assertTrue(entry['languages'])
            self.assertTrue(entry['tags'])
            for flag in ('requires_judge', 'requires_execution'):
                self.assertIsInstance(entry[flag], bool)
                if entry[flag]:
                    if not entry.get('environment_installable'):
                        self.assertTrue(unavailable_reason(entry, ROOT))
        self.assertEqual({item['id'] for item in BENCHMARK_PRESETS},
                         {'smoke', 'daily', 'long_context', 'coding', 'advanced'})
        for preset in BENCHMARK_PRESETS:
            self.assertTrue(preset['label'])
            self.assertTrue(preset['description'])
            self.assertTrue(preset['datasets'])
            self.assertTrue(set(preset['datasets']).issubset(entries))
            self.assertEqual(len(preset['datasets']), len(set(preset['datasets'])))
        for name in ('ruler_32k_gen', 'ruler_128k_gen', 'ruler_256k_gen'):
            self.assertIn('tokenizer', unavailable_reason(entries[name], ROOT))
        self.assertEqual(entries['bfcl_v4_external']['integration'], 'external')
        self.assertIn('未适配', unavailable_reason(entries['bfcl_v4_external'], ROOT))

    def test_new_historical_abbreviations_have_correct_categories_and_variants(self):
        cases = {
            'mmlu_pro_math': ('general_reasoning', 'mmlu_pro'),
            'GPQA_diamond': ('science_reasoning', 'gpqa'),
            'ceval-computer_network': ('chinese_knowledge', 'ceval'),
            'math-500': ('math', 'math500'), 'aime2024': ('math', 'aime'),
            'IFBench': ('instruction', 'ifbench'),
            'LongBenchv2': ('long_context', 'longbench_v2'),
            'chinese_simpleqa': ('factuality', 'simpleqa'),
            'simpleqa': ('factuality', 'simpleqa'),
            'humaneval_plus': ('code', 'code'), 'mbpp_plus': ('code', 'code'),
            'lcb_code_generation_repeat_6': ('code', 'livecodebench'),
            'ruler_niah_single_1_256k': ('long_context', 'ruler'),
            'BFCL_v4': ('tools', 'bfcl'),
        }
        for name, (category, family) in cases.items():
            with self.subTest(name=name):
                explanation = describe_benchmark(name)
                self.assertTrue(explanation['known'])
                self.assertEqual(explanation['category_id'], category)
                self.assertEqual(explanation['family'], family)
                self.assertEqual(explanation['dataset'], name)
                self.assertTrue(explanation['purpose'])
        self.assertEqual(describe_benchmark('chinese_simpleqa')['languages'], ['zh'])
        self.assertIn('evalplus', describe_benchmark('humaneval_plus')['source_url'])
        self.assertIn('evalplus', describe_benchmark('mbpp_plus')['source_url'])

    def test_ifbench_checks_extra_nltk_resources_before_claiming_readiness(self):
        entry = next(item for item in DATASET_CATALOG if item['id'] == 'IFBench_gen')
        with patch('benchmark_catalog.importlib.util.find_spec', return_value=object()), \
                patch('nltk.data.load', return_value=object()), \
                patch('nltk.data.find', side_effect=LookupError('missing')):
            self.assertIn('averaged_perceptron_tagger_eng', unavailable_reason(entry, ROOT))

    def test_cmmlu_subject_names_match_the_installed_official_metadata(self):
        config_root = next((ROOT / '.venv/lib').glob(
            'python*/site-packages/opencompass/configs/datasets'))
        config = Config.fromfile(str(next(config_root.rglob('cmmlu_gen_c13365.py'))))
        for subject, chinese_name in config.cmmlu_subject_mapping.items():
            dataset = 'cmmlu_gen__cmmlu-' + subject
            with self.subTest(subject=subject):
                explanation = describe_benchmark(dataset)
                self.assertEqual(explanation['subject_label'], chinese_name)
                self.assertIn(chinese_name, explanation['label'])
                self.assertIn(chinese_name, explanation['purpose'])
                self.assertIn(chinese_name, explanation['task_example'])
                self.assertEqual(explanation['dataset'], dataset)

    def test_old_unprefixed_results_are_classified_and_demo_is_qualified(self):
        cases = {'demo_gsm8k': 'gsm8k', 'demo math': 'math',
                 'demo_cmmlu-agronomy': 'cmmlu',
                 'lukaemon_mmlu_college_biology': 'mmlu',
                 'bbh-temporal_sequences': 'bbh', 'IFEval': 'ifeval',
                 'LongBench_multifieldqa_zh': 'longbench_qa_zh',
                 'LongBench passage retrieval zh': 'longbench_retrieval_zh',
                 'truthful_qa': 'truthfulqa', 'openai_humaneval_cn': 'code'}
        for name, family in cases.items():
            with self.subTest(name=name):
                result = describe_benchmark(name)
                self.assertTrue(result['known'])
                self.assertEqual(result['family'], family)
                self.assertIn('不能由一个分数直接确认', result['interpretation_note'])
                if name.startswith('demo'):
                    self.assertIn('Demo', result['limitations'])

    def test_unknown_does_not_invent_a_benchmark_or_diagnosis(self):
        result = describe_benchmark('unrecognized custom score')
        self.assertFalse(result['known'])
        self.assertEqual(result['source_url'], '')
        self.assertIn('尚不能', result['impacts'][0])
        self.assertIn('尚未登记', result['purpose'])
        self.assertIn('尚无', result['task_example'])

    def test_subject_results_explain_the_relevant_domain(self):
        medical = describe_benchmark('cmmlu_gen__cmmlu-clinical_medicine')
        law = describe_benchmark('mmlu gen  lukaemon mmlu professional law')
        self.assertIn('医学', medical['focuses'][0])
        self.assertIn('法律', law['focuses'][0])
        self.assertIn('医学', medical['impacts'][0])
        self.assertIn('法律', law['impacts'][0])

    def test_new_supported_configs_serialize_in_one_explicit_config(self):
        selected = [item['id'] for item in DATASET_CATALOG[7:]
                    if not unavailable_reason(item, ROOT)]
        settings = dict(model='test-model', api_key='test-key', base_url='http://127.0.0.1:1/v1')
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / 'eval.py'
            manifest = build_evaluation_config(ROOT, selected, settings, destination)
            config = Config.fromfile(str(destination))
            names = [item['abbr'] for item in config.datasets]
            self.assertEqual(len(names), len(set(names)))
            self.assertEqual([item['id'] for item in manifest], selected)
            self.assertEqual(set(names), {name for item in manifest for name in item['abbrs']})
            self.assertTrue(all(describe_benchmark(name)['known'] for name in names))

    def test_runtime_dependency_and_local_data_checks(self):
        human = next(item for item in DATASET_CATALOG if item['id'] == 'humaneval_cn_gen')
        # Inspect package/data detection independently of the live execution guard.
        with patch('code_runtime.code_readiness', return_value='代码执行环境未就绪'):
            self.assertIn('执行环境', unavailable_reason(human, ROOT))
        human = dict(human)
        human['environment_installable'] = False
        human['required_modules'] = ['human_eval']
        with tempfile.TemporaryDirectory() as tmp, \
                patch('benchmark_catalog.importlib.util.find_spec', return_value=None):
            self.assertIn('human_eval', unavailable_reason(human, Path(tmp)))
        with tempfile.TemporaryDirectory() as tmp, \
                patch('benchmark_catalog.importlib.util.find_spec', return_value=object()), \
                patch('benchmark_catalog.Path.home', return_value=Path(tmp)), \
                patch.dict('benchmark_catalog.os.environ', {'COMPASS_DATA_CACHE': tmp}):
            self.assertIn('数据', unavailable_reason(human, Path(tmp)))
            local = Path(tmp) / human['required_local_files'][0]
            local.parent.mkdir(parents=True)
            local.write_text('', encoding='utf-8')
            self.assertEqual(unavailable_reason(human, Path(tmp)), '')

    def test_missing_auto_download_data_is_declared_without_downloading(self):
        item = next(item for item in DATASET_CATALOG if item['id'] == 'mmlu_gen')
        with tempfile.TemporaryDirectory() as tmp, \
                patch('benchmark_catalog.Path.home', return_value=Path(tmp)), \
                patch.dict('benchmark_catalog.os.environ', {'COMPASS_DATA_CACHE': tmp}):
            status = dataset_preparation(item, Path(tmp))
            self.assertFalse(status['data_cached'])
            self.assertIn('首次开始评测', status['preparation_note'])
            self.assertEqual(list(Path(tmp).iterdir()), [])


if __name__ == '__main__':
    unittest.main()
