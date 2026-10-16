import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from datasets import Dataset
from mmengine.config import Config
from opencompass.registry import ICL_EVALUATORS

from code_evaluation import DockerHumanEvalEvaluator, DockerMBPPEvaluator
from code_runtime import CodeEnvironmentError
from evaluation_config import build_evaluation_config


ROOT = Path(__file__).resolve().parents[1]
SETTINGS = dict(model='test-model', api_key='test-key', base_url='http://127.0.0.1:1/v1')


class DockerCodeEvaluationTests(unittest.TestCase):
    def test_humaneval_uses_matching_real_problem_and_preserves_completion_indent(self):
        samples = [dict(task_id='HumanEval/0', prompt='def f():\n    """说明"""\n',
                        test='def check(candidate):\n    assert candidate() == 1', entry_point='f')]
        with patch('code_evaluation.code_readiness', return_value=''), \
                patch('code_evaluation.execute_case', return_value=dict(passed=True, result='pass')) as execute:
            result = DockerHumanEvalEvaluator().score(['    return 1'], ['HumanEval/0'], samples)
            self.assertEqual(result['humaneval_pass@1'], 100)
            self.assertTrue(result['details']['0']['is_correct'])
            self.assertEqual(execute.call_args.args[2], '    return 1')
            self.assertEqual(execute.call_args.args[3]['test'], samples[0]['test'])
            with self.assertRaises(ValueError):
                DockerHumanEvalEvaluator().score(['return 1'], ['different-id'], samples)

    def test_mbpp_pass_fail_timeout_metrics_and_details(self):
        outcomes = [dict(passed=True, result='pass'), dict(passed=False, result='wrong_answer'),
                    dict(passed=False, result='timeout')]
        with patch('code_evaluation.code_readiness', return_value=''), \
                patch('code_evaluation.execute_case', side_effect=outcomes):
            result = DockerMBPPEvaluator().score(['one', 'two', 'three'], ['assert True'] * 3)
            self.assertAlmostEqual(result['score'], 100 / 3)
            self.assertEqual(result['pass'], 1)
            self.assertEqual(result['wrong_answer'], 1)
            self.assertEqual(result['timeout'], 1)
            self.assertEqual([detail['is_correct'] for detail in result['details'].values()], [True, False, False])

    def test_runtime_failure_is_not_a_zero_model_score(self):
        with patch('code_evaluation.code_readiness', return_value=''), \
                patch('code_evaluation.execute_case', side_effect=CodeEnvironmentError('daemon stopped')):
            with self.assertRaises(CodeEnvironmentError):
                DockerMBPPEvaluator().score(['candidate'], ['assert True'])

    def test_mbpp_preserves_dataset_test_setup_for_container_execution(self):
        sample = dict(test_setup_code='root = Node(1)')
        with patch('code_evaluation.code_readiness', return_value=''), \
                patch('code_evaluation.execute_case', return_value=dict(passed=True, result='pass')) as run:
            DockerMBPPEvaluator().score(['class Node: pass'], ['assert root.value == 1'], [sample])
            self.assertEqual(run.call_args.args[3], dict(tests='assert root.value == 1', setup='root = Node(1)'))

    def test_base_evaluator_result_format_stays_compatible(self):
        dataset = Dataset.from_list([dict(idx=0, subdivision='mbpp')])
        with patch('code_evaluation.code_readiness', return_value=''), \
                patch('code_evaluation.execute_case', return_value=dict(passed=True, result='pass')):
            result = DockerMBPPEvaluator().evaluate(k=[1], n=1, original_dataset=dataset,
                                                  predictions=['candidate'], references=['assert True'])
            self.assertEqual(result['score'], 100)
            self.assertEqual(result['details'][0]['is_correct'], [True])

    def test_generated_config_only_uses_docker_adapters_and_single_candidates(self):
        with tempfile.TemporaryDirectory() as tmp, patch('evaluation_config.code_readiness', return_value=''):
            path = Path(tmp) / 'eval.py'
            build_evaluation_config(ROOT, ['humaneval_cn_gen', 'mbpp_gen'], SETTINGS, path)
            config = Config.fromfile(str(path))
            for dataset, expected in zip(config.datasets, (DockerHumanEvalEvaluator, DockerMBPPEvaluator)):
                self.assertEqual(ICL_EVALUATORS.get(dataset.eval_cfg.evaluator.type), expected)
                self.assertNotIn('pred_postprocessor', dataset.eval_cfg)
                self.assertEqual(dataset.k, [1])
                self.assertEqual(dataset.n, 1)
            for unsafe in ('humaneval_plus_gen', 'mbpp_plus_gen', 'livecodebench_v6_academic'):
                with self.assertRaisesRegex(ValueError, '尚未适配'):
                    build_evaluation_config(ROOT, [unsafe], SETTINGS, path)


if __name__ == '__main__':
    unittest.main()
