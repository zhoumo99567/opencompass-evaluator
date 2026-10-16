"""OpenCompass adapters: all generated code runs in the Docker runner."""
from __future__ import annotations

from pathlib import Path

from opencompass.openicl.icl_evaluator import BaseEvaluator

from code_runtime import CodeEnvironmentError, code_readiness, execute_case


class _DockerEvaluator(BaseEvaluator):
    dataset_id = ''
    kind = ''

    def __init__(self, root=None, timeout_seconds=3):
        super().__init__()
        self.root = Path(root) if root else Path(__file__).resolve().parent
        if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)) or not 0 < timeout_seconds <= 10:
            raise ValueError('代码评分超时必须在 0–10 秒内')
        self.timeout_seconds = timeout_seconds

    def _ready(self):
        reason = code_readiness(self.root, self.dataset_id)
        if reason:
            raise CodeEnvironmentError(reason)

    @staticmethod
    def _single(prediction):
        if isinstance(prediction, list):
            if len(prediction) != 1:
                raise ValueError('当前隔离评分仅支持每题一个候选，不能生成多候选 pass@k')
            prediction = prediction[0]
        if not isinstance(prediction, str):
            raise ValueError('代码候选必须是文本')
        return prediction

    def _execute(self, prediction, problem):
        return execute_case(self.root, self.kind, prediction, problem, self.timeout_seconds)


class DockerHumanEvalEvaluator(_DockerEvaluator):
    dataset_id = 'humaneval_cn_gen'
    kind = 'humaneval_cn'

    def score(self, predictions, references, test_set):
        if len(predictions) != len(references) or len(test_set) != len(predictions) or not predictions:
            raise ValueError('HumanEval 预测、参考及题目数量必须一致且非空')
        self._ready()
        details = {}
        for index, (prediction, reference, sample) in enumerate(zip(predictions, references, test_set)):
            if str(sample.get('task_id')) != str(reference):
                raise ValueError('HumanEval 题目与参考 task_id 不一致')
            problem = {key: sample.get(key) for key in ('prompt', 'test', 'entry_point')}
            if any(not isinstance(value, str) or not value.strip() for value in problem.values()):
                raise ValueError('HumanEval 数据必须包含真实 prompt/test/entry_point')
            candidate = self._single(prediction)
            outcome = self._execute(candidate, problem)
            details[str(index)] = dict(task_id=reference, completion=candidate,
                                       prompt=sample['prompt'], result=outcome['result'],
                                       passed=outcome['passed'], is_correct=outcome['passed'],
                                       execution_image=outcome.get('image_id', ''),
                                       error_type=outcome.get('error_type', ''))
        correct = sum(item['passed'] for item in details.values())
        return {'humaneval_pass@1': correct / len(details) * 100, 'details': details}


class DockerMBPPEvaluator(_DockerEvaluator):
    dataset_id = 'mbpp_gen'
    kind = 'mbpp'

    def score(self, predictions, references, test_set=None):
        if len(predictions) != len(references) or not predictions:
            raise ValueError('MBPP 预测与测试数量必须一致且非空')
        if test_set is not None and len(test_set) != len(predictions):
            raise ValueError('MBPP 题目与预测数量不一致')
        self._ready()
        counts = dict(pass_count=0, timeout=0, failed=0, wrong_answer=0)
        details = {}
        for index, (prediction, tests) in enumerate(zip(predictions, references)):
            if not isinstance(tests, str) or not tests.strip():
                raise ValueError('MBPP 参考必须包含真实测试断言')
            setup = test_set[index].get('test_setup_code', '') if test_set is not None else ''
            if not isinstance(setup, str):
                raise ValueError('MBPP 测试初始化代码必须为文本')
            candidate = self._single(prediction)
            outcome = self._execute(candidate, dict(tests=tests, setup=setup))
            result = outcome['result']
            counts['pass_count' if result == 'pass' else result if result in counts else 'failed'] += 1
            details[str(index)] = dict(origin=candidate, result=result,
                                       execution_image=outcome.get('image_id', ''),
                                       is_correct=outcome['passed'], error_type=outcome.get('error_type', ''))
        passed = counts.pop('pass_count')
        return {'score': passed / len(details) * 100, 'pass': passed,
                **counts, 'details': details}
