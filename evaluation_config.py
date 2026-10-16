"""Build explicit OpenCompass configs; positional configs override CLI datasets."""
from copy import deepcopy
from pathlib import Path
from benchmark_catalog import DATASET_CATALOG
from code_runtime import DATASETS as CODE_DATASETS, code_readiness


def validate_evaluation_options(options):
    """Validate explicit token budgets without coercing bools or decimals."""
    if options is None:
        return None
    if not isinstance(options, dict):
        raise ValueError('评测参数必须是包含上下文长度和输出长度的对象')
    limits = {'max_seq_len': (64, 262144), 'max_out_len': (64, 65536)}
    labels = {'max_seq_len': '上下文长度', 'max_out_len': '输出长度'}
    validated = {}
    for key, (minimum, maximum) in limits.items():
        value = options.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
            raise ValueError(f'{labels[key]}必须是 {minimum}–{maximum} 之间的整数')
        validated[key] = value
    if validated['max_out_len'] >= validated['max_seq_len']:
        raise ValueError('输出长度必须小于上下文长度，需为输入预留空间')
    return validated


def build_evaluation_config(root, selected, settings, destination, options=None):
    options = validate_evaluation_options(options)
    from mmengine.config import Config

    root = Path(root)
    dataset_root = next((root / '.venv/lib').glob(
        'python*/site-packages/opencompass/configs/datasets'))
    base = Config.fromfile(str(root / 'lan_eval.py'))
    models = deepcopy(base.models)
    for model in models:
        model.update(abbr=settings['model'], path=settings['model'],
                     key=settings['api_key'],
                     openai_api_base=settings['base_url'].rstrip('/'))
        if not model['openai_api_base'].endswith('/chat/completions'):
            model['openai_api_base'] += '/chat/completions'
        if options is not None:
            # Fail on excessive input instead of silently dropping evidence
            # from long-context benchmarks.
            model.update(**options, mode='none')
    datasets = []
    manifest = []
    catalog = {entry['id']: entry for entry in DATASET_CATALOG}
    for dataset_id in selected:
        if catalog.get(dataset_id, {}).get('requires_execution'):
            if dataset_id not in CODE_DATASETS:
                raise ValueError('此代码评测尚未适配 Docker 隔离评分，不能使用宿主机评分器')
            reason = code_readiness(root, dataset_id)
            if reason:
                raise ValueError(reason)
        matches = list(dataset_root.rglob(dataset_id + '.py'))
        if len(matches) != 1:
            raise ValueError(f'无法唯一定位评测集：{dataset_id}')
        config = Config.fromfile(str(matches[0]))
        items = [deepcopy(item) for key in config.keys()
                 if key.endswith('_datasets') for item in config[key]]
        if not items:
            raise ValueError(f'评测配置不包含数据集：{dataset_id}')
        abbrs = []
        for item in items:
            if dataset_id in CODE_DATASETS:
                adapter = ('DockerHumanEvalEvaluator' if dataset_id == 'humaneval_cn_gen'
                           else 'DockerMBPPEvaluator')
                item['eval_cfg']['evaluator'] = dict(
                    type='code_evaluation.' + adapter, root=str(root),
                    timeout_seconds=3 if dataset_id == 'humaneval_cn_gen' else 10)
                # Preserve raw indentation and extract code only inside Docker.
                item['eval_cfg'].pop('pred_postprocessor', None)
                item['k'] = [1]
                item['n'] = 1
            if options is not None:
                item.setdefault('infer_cfg', {}).setdefault('inferencer', {})[
                    'max_out_len'] = options['max_out_len']
            # Chat and Base variants often share an upstream abbreviation.
            # A stable prefix keeps their prediction/result files independent.
            item['abbr'] = f"{dataset_id}__{item['abbr']}"
            abbrs.append(item['abbr'])
            datasets.append(item)
        manifest.append({'id': dataset_id, 'abbrs': abbrs})
    names = [item['abbr'] for item in datasets]
    if len(names) != len(set(names)):
        raise ValueError('评测配置含有重复结果名称')
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    Config(dict(models=models, datasets=datasets)).dump(str(destination))
    destination.chmod(0o600)
    return manifest
