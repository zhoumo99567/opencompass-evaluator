"""Curated OpenCompass configs and conservative explanations of their scores.

The entries describe evaluation tasks, not evidence about a model's training
data. A score alone cannot establish why a model failed.
"""

from __future__ import annotations

from copy import deepcopy
import importlib.util
import os
from pathlib import Path
import re


INTERPRETATION_NOTE = (
    '低分原因仅为排查假设，不能由一个分数直接确认。请结合错误样本、输出是否截断、'
    '答案提取规则和接口失败记录判断；业务影响是任务相关性说明，不代表实际业务事故率。'
)

BENCHMARK_CATEGORIES = [
    dict(id='chinese_knowledge', label='中文知识与理解', description='中文学科知识、题意理解与专业概念。', order=10),
    dict(id='general_reasoning', label='综合知识与推理', description='跨学科英文知识与复杂选项辨析。', order=20),
    dict(id='science_reasoning', label='高难科学推理', description='研究生级科学问题，观察专业推理上限。', order=30),
    dict(id='math', label='数学计算与解题', description='文字算术、符号运算和竞赛数学。', order=40),
    dict(id='logic', label='逻辑与规则推理', description='多步逻辑、条件判断和状态追踪。', order=50),
    dict(id='instruction', label='指令与格式遵循', description='关键词、长度、结构等可验证的输出约束。', order=60),
    dict(id='code', label='代码与编程', description='函数或程序生成、测试通过率和执行结果预测。', order=70),
    dict(id='long_context', label='长上下文理解', description='长文问答、证据定位和不同长度下的有效理解。', order=80),
    dict(id='factuality', label='事实可靠性', description='短事实问答、错误前提和常见误解处理。', order=90),
    dict(id='tools', label='工具调用与 Agent', description='函数选择、调用参数和多步工具交互。', order=100),
]
_CATEGORY_LABELS = {item['id']: item['label'] for item in BENCHMARK_CATEGORIES}
_FAMILY_CATEGORIES = {
    'gsm8k': 'math', 'math': 'math', 'math500': 'math', 'aime': 'math',
    'cmmlu': 'chinese_knowledge', 'ceval': 'chinese_knowledge',
    'mmlu': 'general_reasoning', 'mmlu_pro': 'general_reasoning',
    'gpqa': 'science_reasoning', 'bbh': 'logic',
    'ifeval': 'instruction', 'ifbench': 'instruction',
    'code': 'code', 'evalplus': 'code', 'livecodebench': 'code',
    'longbench_qa_zh': 'long_context', 'longbench_retrieval_zh': 'long_context',
    'longbench_v2': 'long_context', 'ruler': 'long_context',
    'truthfulqa': 'factuality', 'simpleqa': 'factuality', 'bfcl': 'tools',
}
_FAMILY_LANGUAGES = {
    'cmmlu': ['zh'], 'ceval': ['zh'], 'longbench_qa_zh': ['zh'],
    'longbench_retrieval_zh': ['zh'], 'longbench_v2': ['en'],
}

_SUBJECT_DOMAINS = (
    (('anatomy', 'medicine', 'medical', 'clinical', 'virology', 'nutrition',
      'genetics', 'human sexuality'),
     '医学与健康知识', ['医学资料中的概念理解', '健康科普知识问答', '相关专业学习辅助']),
    (('law', 'jurisprudence'), '法律知识',
     ['法律概念辨析', '法律资料阅读', '相关专业学习辅助']),
    (('economics', 'economic', 'accounting', 'business', 'finance', 'management', 'marketing'),
     '经济与商业知识', ['经济与商业资料分析', '财务概念问答', '相关专业学习辅助']),
    (('computer', 'machine learning'), '计算机知识',
     ['技术资料理解', '计算机基础问答', '编程与系统设计的知识辅助']),
    (('mathematics', 'math', 'statistics', 'electrical engineering', 'physics', 'chemistry'),
     '理工科知识', ['理工科资料理解', '公式与技术概念辨析', '相关专业学习辅助']),
    (('history', 'literature', 'arts', 'culture', 'philosophy', 'religion'),
     '人文知识', ['历史与文化知识问答', '人文资料阅读', '相关内容写作辅助']),
)

# CMMLU labels come from the installed upstream cmmlu_subject_mapping;
# additional MMLU names use straightforward Chinese subject translations.
_SUBJECT_LABELS = {
    'agronomy': '农学', 'anatomy': '解剖学', 'ancient_chinese': '古汉语',
    'arts': '艺术学', 'astronomy': '天文学', 'business_ethics': '商业伦理',
    'chinese_civil_service_exam': '中国公务员考试', 'chinese_driving_rule': '中国驾驶规则',
    'chinese_food_culture': '中国饮食文化', 'chinese_foreign_policy': '中国外交政策',
    'chinese_history': '中国历史', 'chinese_literature': '中国文学',
    'chinese_teacher_qualification': '中国教师资格', 'clinical_knowledge': '临床知识',
    'college_actuarial_science': '大学精算学', 'college_education': '大学教育学',
    'college_engineering_hydrology': '大学工程水文学', 'college_law': '大学法律',
    'college_mathematics': '大学数学', 'college_medical_statistics': '大学医学统计',
    'college_medicine': '大学医学', 'computer_science': '计算机科学',
    'computer_security': '计算机安全', 'conceptual_physics': '概念物理学',
    'construction_project_management': '建设工程管理', 'economics': '经济学',
    'education': '教育学', 'electrical_engineering': '电气工程',
    'elementary_chinese': '小学语文', 'elementary_commonsense': '小学常识',
    'elementary_information_and_technology': '小学信息技术',
    'elementary_mathematics': '初等数学', 'ethnology': '民族学',
    'food_science': '食品科学', 'genetics': '遗传学', 'global_facts': '全球事实',
    'high_school_biology': '高中生物', 'high_school_chemistry': '高中化学',
    'high_school_geography': '高中地理', 'high_school_mathematics': '高中数学',
    'high_school_physics': '高中物理学', 'high_school_politics': '高中政治',
    'human_sexuality': '人类性行为', 'international_law': '国际法学',
    'journalism': '新闻学', 'jurisprudence': '法理学',
    'legal_and_moral_basis': '法律与道德基础', 'logical': '逻辑学',
    'machine_learning': '机器学习', 'management': '管理学', 'marketing': '市场营销',
    'marxist_theory': '马克思主义理论', 'modern_chinese': '现代汉语',
    'nutrition': '营养学', 'philosophy': '哲学',
    'professional_accounting': '专业会计', 'professional_law': '专业法学',
    'professional_medicine': '专业医学', 'professional_psychology': '专业心理学',
    'public_relations': '公共关系', 'security_study': '安全研究',
    'sociology': '社会学', 'sports_science': '体育学',
    'traditional_chinese_medicine': '中医中药', 'virology': '病毒学',
    'world_history': '世界历史', 'world_religions': '世界宗教',
    'abstract_algebra': '抽象代数', 'college_biology': '大学生物',
    'college_chemistry': '大学化学', 'college_computer_science': '大学计算机科学',
    'college_physics': '大学物理', 'econometrics': '计量经济学',
    'formal_logic': '形式逻辑', 'high_school_computer_science': '高中计算机科学',
    'high_school_european_history': '高中欧洲历史',
    'high_school_government_and_politics': '高中政府与政治',
    'high_school_macroeconomics': '高中宏观经济', 'high_school_microeconomics': '高中微观经济',
    'high_school_psychology': '高中心理学', 'high_school_statistics': '高中统计',
    'high_school_us_history': '高中美国历史', 'high_school_world_history': '高中世界历史',
    'human_aging': '人类衰老', 'logical_fallacies': '逻辑谬误',
    'medical_genetics': '医学遗传学', 'miscellaneous': '综合知识',
    'moral_disputes': '道德争议', 'moral_scenarios': '道德情境',
    'prehistory': '史前史',
    'us_foreign_policy': '美国外交政策',
    'math': '数学', 'physics': '物理', 'chemistry': '化学', 'biology': '生物',
    'law': '法律', 'engineering': '工程', 'history': '历史',
    'psychology': '心理学', 'business': '商业', 'health': '健康', 'other': '其他学科',
    'computer_network': '计算机网络', 'operating_system': '操作系统',
    'computer_architecture': '计算机组成', 'accountant': '会计',
    'advanced_mathematics': '高等数学', 'probability_and_statistics': '概率统计',
    'discrete_mathematics': '离散数学', 'civil_servant': '公务员',
}

_PURPOSES = {
    'gsm8k': (
        '检查模型能否读懂数量关系，并连续计算得到正确的最终答案。',
        '每盒彩笔 12 元，买 3 盒付 50 元，应找回多少钱？',
        ['日常数量与费用问答', '多步计算辅助']),
    'math': (
        '检查模型面对较难数学题时的解题、符号运算和最终答案准确性。',
        '已知一个二次方程的两根满足给定条件，求参数的取值范围。',
        ['复杂数学答疑', '公式和定量推导辅助']),
    'cmmlu': (
        '检查模型能否用中文理解学科问题，并从相近选项中选出正确知识。',
        '植物进行光合作用时主要吸收哪种气体？从四个选项中选择。',
        ['中文知识助手', '按学科选择模型']),
    'mmlu': (
        '检查模型对英文多学科知识的掌握，以及专业概念和选项的辨析能力。',
        '阅读一道英文生物或经济学题，从 A/B/C/D 中选出正确答案。',
        ['英文专业问答', '跨学科资料理解']),
    'bbh': (
        '检查模型能否遵守题目规则，连续推理并跟踪中间状态。',
        '三个人依次交换手中的物品，问最后谁拿着最初的红球。',
        ['多条件规则判断', '流程和状态推理辅助']),
    'ifeval': (
        '检查模型是否真的遵守字数、关键词、句数和格式等明确要求。',
        '用英文介绍海洋：必须包含 ocean，且恰好写 3 句话。',
        ['规定格式的批量内容', '带多项输出约束的流程']),
    'longbench_qa_zh': (
        '检查模型读过长篇中文资料后，能否找到依据并简短回答问题。',
        '提供一篇长报告，询问其中某项计划的负责人和启动年份。',
        ['长报告问答', '给定资料的事实提取']),
    'longbench_retrieval_zh': (
        '检查模型能否在很多已提供的段落中找到目标信息所在的位置。',
        '给出编号段落和一段内容摘要，只回答与摘要对应的段落编号。',
        ['长文档证据定位', '引用段落查找']),
    'truthfulqa': (
        '检查模型面对常见误解或错误前提时，是否仍能给出真实回答。',
        '被问“为什么所有蛇都有毒？”时，先检查这个前提是否成立。',
        ['易误导问题的事实核查', '知识问答可靠性对照']),
    'code': (
        '检查模型写出的 Python 小函数是否满足要求，并能通过测试。',
        '实现一个函数：保留列表中偶数的顺序，并正确处理空列表。',
        ['函数级代码生成', '小型脚本辅助']),
}

_PROFILES = {
    'gsm8k': {
        'label': 'GSM8K', 'category': '数学',
        'focuses': ['多步算术推理', '文字题理解', '数量关系与单位'],
        'metric_meaning': 'accuracy：生成答案经提取后与标准数值答案匹配的比例，通常为 0–100%。',
        'low_score_reasons': ['可能误解题目条件或数量关系', '可能在多步计算中出错', '可能未输出可提取的最终数值，或推理输出被截断'],
        'impacts': ['带金额、数量、比例的问答', '预算与表格计算辅助', '分步骤规划中的数值可靠性'],
        'limitations': '测试小学水平的英文数学文字题；得分不等于财务任务准确率，且只核对最终答案，不能证明推理过程正确。',
        'source_url': 'https://github.com/openai/grade-school-math',
    },
    'math': {
        'label': 'MATH', 'category': '数学',
        'focuses': ['竞赛数学', '代数与几何', '符号表达和复杂推理'],
        'metric_meaning': 'accuracy：提取并规范化后的最终数学答案与参考答案等价的比例，通常为 0–100%。',
        'low_score_reasons': ['可能缺少对应数学知识或解题策略', '可能在长推理或符号运算中出错', '可能答案格式不符合提取规则，或输出长度不足'],
        'impacts': ['复杂数学答疑', '公式推导与符号计算辅助', '需要严格计算的科学分析'],
        'limitations': '竞赛题与日常业务难度不同；当前经典配置采用答案规范化规则，并非对解题过程逐步证明。不同提示词和评分器结果不可直接混比。',
        'source_url': 'https://github.com/hendrycks/math',
    },
    'cmmlu': {
        'label': 'CMMLU', 'category': '知识',
        'focuses': ['中文多学科知识', '中文题干理解', '中国语境下的知识与推理'],
        'metric_meaning': 'accuracy：各学科选择题答案匹配率；完整配置含 67 个学科，应结合分学科结果，通常为 0–100%。',
        'low_score_reasons': ['可能对应学科知识覆盖不足', '可能中文概念、题意或选项辨析错误', '可能输出额外文字导致选项提取失败，或提示方式与模型不匹配'],
        'impacts': ['中文知识助手', '相应学科的教育与问答', '中文专业资料理解'],
        'limitations': '选择题识别正确选项不等于自由回答可靠；学科平均值可能掩盖弱项，不能替代实际业务题和专业审核。',
        'source_url': 'https://github.com/haonan-li/CMMLU',
    },
    'mmlu': {
        'label': 'MMLU', 'category': '知识',
        'focuses': ['英文多学科知识', '专业概念辨析', '学科推理'],
        'metric_meaning': 'accuracy：各学科选择题答案匹配率；完整配置含 57 个学科，应结合分学科结果，通常为 0–100%。',
        'low_score_reasons': ['可能专业知识覆盖不足', '可能英文题意或近似选项辨析错误', '可能题干提示或答案格式不适配，导致选项提取错误'],
        'impacts': ['英文知识问答', '专业资料阅读', '多学科学习辅助'],
        'limitations': '这是封闭式英文考试题；总分不能直接代表中文能力、工具使用能力或某个专业场景的可用性。',
        'source_url': 'https://github.com/hendrycks/test',
    },
    'bbh': {
        'label': 'BBH', 'category': '推理',
        'focuses': ['逻辑推理', '多步状态追踪', '符号、算术与语言推理'],
        'metric_meaning': '各子任务的答案匹配正确率，通常为 0–100%。BBH 原始定义含 23 类任务；当前配置按逻辑演绎与物体追踪的题目规模拆分为 27 个子任务。应先看子任务，再看汇总。',
        'low_score_reasons': ['可能无法稳定跟踪中间状态或长推理链', '可能不熟悉任务规则或少样本示例', '可能最终答案格式不符合该子任务的匹配规则'],
        'impacts': ['复杂规则判断', '流程与多步计划辅助', '需要保持状态一致的逻辑任务'],
        'limitations': '多个任务是受控或合成题；提示词是否包含推理示例会影响分数，不能直接推断真实 Agent 成功率。',
        'source_url': 'https://github.com/suzgunmirac/BIG-Bench-Hard',
    },
    'ifeval': {
        'label': 'IFEval', 'category': '指令遵循',
        'focuses': ['可验证的输出约束', '格式与长度要求', '多条件指令遵循'],
        'metric_meaning': 'Prompt-level strict 要求一条提示的全部约束都通过；Inst-level strict 按单条约束统计。loose 放宽部分格式检查，四项均为 0–100%。',
        'low_score_reasons': ['可能遗漏某条约束或无法同时满足多条件', '可能增加了要求外的解释、标题或 Markdown', '可能思考内容混入正文、输出截断或严格检查器存在语言边界'],
        'impacts': ['规定格式的内容生成', '需要稳定输出约束的自动流程', '批量文案的格式与长度控制'],
        'limitations': '只覆盖可程序验证的一组约束；不等于内容事实正确或所有 JSON/工具调用都可靠。strict 与 loose 必须分别解读。',
        'source_url': 'https://github.com/google-research/google-research/tree/master/instruction_following_eval',
    },
    'longbench_qa_zh': {
        'label': 'LongBench 中文长文问答', 'category': '长上下文',
        'focuses': ['中文长文理解', '长上下文中的事实定位', '基于原文的简短回答'],
        'metric_meaning': '中文答案 F1：模型简短答案与参考答案的词项重合程度，汇总通常为 0–100；并非逐题完全答对的比例。',
        'low_score_reasons': ['可能在长文中遗漏或混淆关键事实', '可能输入被客户端或服务端截断', '可能答案过长、表达与参考不同或生成被截断，从而降低词项 F1'],
        'impacts': ['长报告与合同阅读', '长资料问答', '大上下文 RAG 中对给定资料的回答质量'],
        'limitations': '采用 LongBench v1 中文 multifieldqa 子集；本身不能验证 256k 全窗口能力。F1 受答案措辞影响，也不评估检索器是否找到了正确文档。',
        'source_url': 'https://github.com/THUDM/LongBench/tree/main/LongBench',
    },
    'longbench_retrieval_zh': {
        'label': 'LongBench 中文段落检索', 'category': '长上下文',
        'focuses': ['长文本中的目标段落定位', '相似段落辨析', '长上下文注意力与检索'],
        'metric_meaning': 'Retrieval score：从输出提取所有数字，计算目标段落编号在这些数字中的匹配比例，再按题目平均为 0–100。只返回正确编号得满分，额外数字可能降低得分。',
        'low_score_reasons': ['可能在长上下文中没有定位到目标段落', '可能被相似段落干扰', '可能输入截断，或输出解释中出现额外数字导致匹配比例降低'],
        'impacts': ['长文档证据定位', '知识库引用定位', '大量上下文中的信息查找'],
        'limitations': '采用 LongBench v1 中文 passage_retrieval 子集，测试模型对已提供文本的定位；不等于外部向量检索性能，也不证明任意长度下均可靠。',
        'source_url': 'https://github.com/THUDM/LongBench/tree/main/LongBench',
    },
    'truthfulqa': {
        'label': 'TruthfulQA', 'category': '事实可靠性',
        'focuses': ['常见误解与错误前提', '事实性回答', '避免模仿错误知识'],
        'metric_meaning': '当前 generation 配置使用 TruthfulQA 裁判模型评分；它与多选 MC1/MC2 或文本相似度指标含义不同，应以实际 metric 为准。',
        'low_score_reasons': ['可能复述了常见错误说法或接受错误前提', '可能缺少相关事实知识或不善于表达不确定性', '可能裁判模型和回答措辞存在偏差，需要人工检查失败样本'],
        'impacts': ['事实性知识问答', '误导性问题的处理', '需要核查来源的用户咨询'],
        'limitations': '只针对一组易诱发错误的题目；不是所有场景的幻觉率，也不能只凭高分承诺事实可靠。当前配置额外依赖本地 7B 裁判模型。',
        'source_url': 'https://github.com/sylinrl/TruthfulQA',
    },
    'code': {
        'label': '代码生成', 'category': '代码',
        'focuses': ['根据要求编写 Python 函数', '功能正确性', '边界条件与测试通过率'],
        'metric_meaning': 'pass@k 表示采样 k 个候选时至少一个通过测试的估计概率；单候选重点看 pass@1。MBPP 一般报告测试通过正确率。不同 k 不可直接混比。',
        'low_score_reasons': ['可能算法、边界处理或库/API 用法错误', '可能遗漏函数签名、导入或代码格式不符合提取规则', '可能执行环境、依赖、超时或评分器出现问题，应先区分执行故障和代码错误'],
        'impacts': ['函数级代码生成', '小型自动化脚本', '编程辅助的首次可运行率'],
        'limitations': '通过有限单元测试不等于工程质量、安全性或真实仓库任务完成率；必须有合适的代码执行环境。',
        'source_url': 'https://github.com/openai/human-eval',
    },
}


def _add_profile(family, label, purpose, example, uses, focuses, metric,
                 checks, limitations, source):
    _PURPOSES[family] = (purpose, example, uses)
    _PROFILES[family] = dict(
        label=label, category=_CATEGORY_LABELS[_FAMILY_CATEGORIES[family]],
        focuses=focuses, metric_meaning=metric, low_score_reasons=checks,
        impacts=list(uses), limitations=limitations, source_url=source)


_add_profile(
    'mmlu_pro', 'MMLU-Pro',
    '检查模型在更多相近选项和较难跨学科题目中，能否结合知识完成推理。',
    '阅读一道专业科学题，从多个相近解释中选择最符合条件的一项。',
    ['跨学科专业问答', '较难知识推理对照'],
    ['跨学科专业知识', '多选项辨析', '知识与推理结合'],
    'accuracy：从生成回答提取选项后，统计匹配正确率，通常为 0–100%；当前配置按 14 个学科分项。',
    ['可能专业知识或多步推理不足', '可能混淆更相近的候选答案', '可能最终选项提取失败或输出被截断'],
    '比 MMLU 的题目和候选选项设置不同，不能直接比较裸分；不是通用日常任务的成功率。',
    'https://github.com/TIGER-AI-Lab/MMLU-Pro')
_add_profile(
    'gpqa', 'GPQA Diamond',
    '用较难的研究生级科学选择题，观察模型的专业知识与科学推理上限。',
    '给出化学或物理条件，判断四个候选机制或结论中哪一个成立。',
    ['高难科学问答对照', '专业推理上限观察'],
    ['物理、化学与生物知识', '研究生级科学推理'],
    'accuracy：生成答案中的 A/B/C/D 选项与参考匹配率，通常为 0–100%；gpqa_gen 实际使用 Diamond 子集。',
    ['可能缺少相关专业知识或推导能力', '可能受近似选项干扰', '可能推理截断或选项提取错误'],
    '这是高难且规模有限的科学题，低分不能直接判定日常助手不可用；需固定 Diamond 子集、提示和采样设置。',
    'https://github.com/idavidrein/gpqa')
_add_profile(
    'ceval', 'C-Eval',
    '检查模型对中文考试和专业学科题的理解，按学科查看知识覆盖。',
    '阅读一道中文计算机网络或法律题，从 A/B/C/D 中选择正确答案。',
    ['中文知识助手选型', '中文专业概念问答'],
    ['中文学科知识', '专业概念辨析', '中文选择题理解'],
    'accuracy：各学科生成选项匹配率，通常为 0–100%；当前 ceval_gen 含 52 个学科，评估 val 集。',
    ['可能相关学科知识不足', '可能误读中文题干或相近概念', '可能答案选项提取失败'],
    '当前使用有公开标签的验证集，不是官方隐藏测试集榜单；与 CMMLU 有重叠，分数不能相互替代。',
    'https://cevalbenchmark.com/')
_add_profile(
    'math500', 'MATH-500',
    '用 MATH 中选出的 500 道题比较数学解题能力，减少完整数学评测的题量。',
    '解一道代数或几何题，并明确给出可验证的最终数学答案。',
    ['中等规模数学回归', '数学解题能力比较'],
    ['竞赛数学', '符号表达', '最终答案正确性'],
    'accuracy：当前配置用 MATHVerifyEvaluator 提取并做数学等价验证，通常为 0–100%。',
    ['可能数学知识或推导不足', '可能符号表达未被评分器正确解析', '可能输出预算不足或最终答案缺失'],
    '这是固定 500 题子集，不是完整 MATH；符号验证器、答案格式和提示不同会影响结果，不验证每一步推理。',
    'https://huggingface.co/datasets/HuggingFaceH4/MATH-500')
_add_profile(
    'aime', 'AIME 2024',
    '用高难竞赛数学题观察模型能否完成较长推导并得到精确最终答案。',
    '根据数论或几何条件求一个整数，并只给出明确的最终结果。',
    ['高难数学上限观察', '推理预算变化对照'],
    ['竞赛数学', '长链推导', '精确整数答案'],
    'accuracy：当前配置用 MATHVerifyEvaluator 判断最终答案等价性，通常为 0–100%；本配置不是重复采样的 pass@k。',
    ['可能缺少竞赛数学策略', '可能长推导中计算或逻辑错误', '可能思考耗尽输出预算，未产生可验证的答案'],
    '年度题集较小，单题会明显影响分数；不能把少量高难题分数当作日常能力总评，重复次数需另行报告。',
    'https://github.com/open-compass/opencompass/tree/main/opencompass/configs/datasets/aime2024')
_add_profile(
    'ifbench', 'IFBench',
    '检查模型面对更多可验证的约束类型时，能否准确遵守指令。',
    '生成指定格式文本，同时满足用词、字符位置和输出结构等约束。',
    ['复杂输出约束回归', '规定格式的生成流程'],
    ['约束组合', '输出结构与字符规则', '指令遵循泛化'],
    '报告 prompt/instance 的 strict 与 loose 正确率；当前 OpenCompass score 是四项正确率的算术平均，范围 0–100%。',
    ['可能遗漏或冲突处理某项约束', '可能增加了要求外的解释或格式', '可能评分依赖、语言或规则边界影响判断'],
    '不是 IFEval 的同一题集，两个分数不能直接横比；规则通过不代表内容真实、工具调用正确或业务流程成功。',
    'https://github.com/allenai/IFBench')
_add_profile(
    'longbench_v2', 'LongBench v2',
    '用真实长资料中的选择题，检查模型能否整合证据并完成长文推理。',
    '提供长篇资料和四个候选结论，选择最符合全文证据的一项。',
    ['复杂长文问答', '跨资料证据整合'],
    ['长资料理解', '跨证据推理', '按长度与难度分项观察'],
    'accuracy：从生成回答提取 A/B/C/D 并核对参考答案，通常为 0–100%；另有长度、难度等分项。',
    ['可能长文事实遗漏或证据整合错误', '可能题目超过实际输入窗口', '可能答案提取或推理输出预算不足'],
    '题目原始长度差异很大，部分超过当前服务窗口；不能静默截断后称完整集成绩，也不是 256k 精确长度测试。',
    'https://github.com/THUDM/LongBench')
_add_profile(
    'simpleqa', 'SimpleQA',
    '检查模型能否简短准确回答有明确事实答案的问题，并区分错误与未作答。',
    '询问某项公开事件的具体年份，要求给出简短事实答案。',
    ['短事实问答可靠性', '知识助手错误与拒答分析'],
    ['事实性回答', '正确、错误与未作答区分'],
    '由独立 LLM 裁判按参考答案判断正确、错误或未作答；具体百分比、分母和派生指标以实际裁判后处理为准。',
    ['可能相关事实知识不足', '可能猜测具体事实或未正确表达不确定性', '可能裁判提示、裁判能力或回答措辞影响分类'],
    '需要另配独立裁判模型，不能默认让被测模型自评；短事实题不等于带检索或所有业务场景的幻觉率。',
    'https://github.com/openai/simple-evals')
_add_profile(
    'livecodebench', 'LiveCodeBench v6',
    '用固定版本的编程任务，检查代码生成、执行结果理解和测试输出预测。',
    '根据题面生成可运行程序，或预测给定程序在测试输入下的输出。',
    ['竞赛代码生成比较', '代码执行与测试理解'],
    ['程序生成', '代码执行推理', '测试输出预测'],
    '按任务报告测试通过率或 pass@1 等指标；不同任务和重复采样应分开，当前 v6 academic 配置含三种任务。',
    ['可能算法、复杂度或边界条件错误', '可能代码提取或输入输出格式不符', '可能执行依赖、超时或环境导致测试失败'],
    '固定 v6 配置不代表最新题目；必须记录数据版本与时间范围，配置执行环境，并区分代码错误与运行基础设施故障。',
    'https://github.com/LiveCodeBench/LiveCodeBench')
_add_profile(
    'ruler', 'RULER',
    '在指定上下文长度下，检查模型能否检索、追踪和整合合成文本中的信息。',
    '在长文中埋入多个键值或变量关系，要求找出目标值并按规则回答。',
    ['有效上下文长度对照', '长输入能力随长度变化分析'],
    ['多针检索', '变量追踪', '聚合与长文问答'],
    '各合成子任务按参考答案的匹配规则计分，通常为 0–100%；必须同时报告任务、实际 tokenizer 和长度。',
    ['可能长输入下遗忘目标或状态追踪错误', '可能实际 tokenizer 长度与目标不符', '可能窗口不足、截断或输出格式不匹配'],
    '当前未配置实际模型 tokenizer 与窗口预算；32k/128k/256k 是目标配置名，不可直接当作服务接受的精确输入长度。输入、输出和模板须共同留在窗口内。',
    'https://github.com/NVIDIA/RULER')
_add_profile(
    'bfcl', 'BFCL 工具调用',
    '检查模型能否选择合适函数、填写参数，并在需要时完成多步工具交互。',
    '根据用户需求，在天气和日历等函数中选择正确函数并生成符合定义的参数。',
    ['工具调用助手选型', '函数参数与多步交互验证'],
    ['函数选择', '参数正确性', '多步工具使用'],
    'BFCL 按版本和任务评估调用结构、参数或执行结果；当前项目尚未适配，不能产生本地有效分数。',
    ['可能函数选择或参数理解错误', '可能工具 schema 或调用格式不兼容', '可能多步交互状态或工具执行环境影响结果'],
    '当前是明确禁用的外部项目占位；需要专用调用适配、工具环境和评分流程，不能仅把聊天文本答案当作工具调用测试。',
    'https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard')


def _entry(dataset_id, label, category, description, family, **extra):
    item = dict(id=dataset_id, label=label, category=category,
                description=description, family=family, **extra)
    profile = deepcopy(_PROFILES[family])
    for key in ('focuses', 'metric_meaning', 'low_score_reasons', 'impacts',
                'limitations', 'source_url'):
        item.setdefault(key, profile[key])
    purpose, example, uses = _PURPOSES[family]
    item.setdefault('purpose', purpose)
    item.setdefault('task_example', example)
    item.setdefault('suitable_uses', list(uses))
    if dataset_id.startswith('demo_'):
        item['purpose'] = (
            '用少量样本检查接口、答案提取和评分流程，同时观察' + {
                'gsm8k': '多步算术是否正确。', 'math': '竞赛数学解题是否正确。',
                'cmmlu': '中文学科选择题是否答对。'}[family])
        item['suitable_uses'] = ['首次验证模型接口', '调整提示词后的快速回归']
    if dataset_id == 'demo_gsm8k_base_gen':
        item['purpose'] = '使用 Base 推断配置，快速检查数学文字题；本安装沿用 Chat 提示，并额外设置停止条件。'
    elif dataset_id == 'demo_math_base_gen':
        item['purpose'] = '使用 Base 少样本提示配置，快速检查竞赛数学解题，便于与 Chat 配置对照。'
    elif dataset_id == 'demo_cmmlu_base_ppl':
        item['purpose'] = '通过比较选项概率检查中文知识选择能力；当前聊天接口不支持此评分方式。'
        item['metric_meaning'] = '先比较候选答案的困惑度并选择最低的选项，再统计选择正确率。报告中的 accuracy 越高越好，内部困惑度越低越好；当前 Chat Completions 适配器不可运行。'
        item['suitable_uses'] = ['支持 token 概率接口时对照选择题评分']
    elif dataset_id == 'humaneval_cn_gen':
        item['purpose'] = '根据中文函数说明生成 Python 代码，检查功能和边界条件是否通过测试。'
        item['suitable_uses'] = ['中文说明的函数生成', 'Python 编程辅助']
    elif dataset_id == 'mbpp_gen':
        item['purpose'] = '根据简短编程要求生成 Python 小函数，再执行测试检查能否正确运行。'
        item['metric_meaning'] = 'score：生成代码通过该题测试的比例，通常为 0–100%。当前 MBPP 配置每题生成一个候选；与多候选 pass@k 分开比较。'
    item['metric_explanation'] = item['metric_meaning']
    item['low_score_checks'] = deepcopy(item['low_score_reasons'])
    item.setdefault('category_id', _FAMILY_CATEGORIES[family])
    item['category'] = _CATEGORY_LABELS[item['category_id']]
    item.setdefault('languages', list(_FAMILY_LANGUAGES.get(family, ['en'])))
    item.setdefault('scope', 'demo' if dataset_id.startswith('demo_') else 'full')
    item.setdefault('requires_judge', family in {'truthfulqa', 'simpleqa'})
    item.setdefault('requires_execution', family in {'code', 'livecodebench'})
    item['environment_installable'] = dataset_id in {'humaneval_cn_gen', 'mbpp_gen'}
    item.setdefault('integration', 'opencompass')
    item.setdefault('tags', [item['category'], '快速验证' if item['scope'] == 'demo' else '完整配置'])
    if item['requires_judge']:
        item.setdefault('unsupported_reason', '未配置独立裁判模型；不能用被测模型自动代替裁判')
    if item['requires_execution']:
        item.setdefault('unsupported_reason', '未配置代码评测的隔离执行环境；评分会执行模型生成的代码')
        if item['environment_installable']:
            item.pop('unsupported_reason', None)
            item['execution_environment'] = 'Docker 隔离 / Python 3.11 标准库 / 每题一个候选'
            item['limitations'] += '当前隔离运行器仅含 Python 3.11 标准库；第三方库需求需另行适配，不能与不同依赖环境的分数直接混比。'
        elif item['integration'] == 'opencompass':
            item['unsupported_reason'] = '此数据集的 Docker 评分适配尚未完成；安装基础环境不会自动启用原宿主机评分器'
    item['requires_packages'] = list(item.get('required_modules', []))
    item['nltk_resources'] = (['tokenizers/punkt', 'tokenizers/punkt_tab/english']
                              if item.get('required_nltk_tokenizer') else []) + list(item.get('required_nltk_resources', []))
    item['interpretation_note'] = INTERPRETATION_NOTE
    if dataset_id.startswith('demo_'):
        item['limitations'] = 'Demo 只适合接口检查与少量样本验证，样本少且不适合与完整榜单比较。' + item['limitations']
    return item


# Preserve the original seven IDs so saved collections and reports keep working.
DATASET_CATALOG = [
    _entry('demo_gsm8k_chat_gen', 'Demo GSM8K', '数学', '小学数学推理，生成式评分', 'gsm8k'),
    _entry('demo_math_chat_gen', 'Demo MATH', '数学', '竞赛数学，生成式评分', 'math'),
    _entry('demo_cmmlu_chat_gen', 'Demo CMMLU', '知识', '中文知识问答，生成式评分', 'cmmlu'),
    _entry('demo_gsm8k_base_gen', 'Demo GSM8K Base', '数学', 'GSM8K 基础配置', 'gsm8k'),
    _entry('demo_math_base_gen', 'Demo MATH Base', '数学', 'MATH 基础配置', 'math'),
    _entry('demo_cmmlu_base_ppl', 'Demo CMMLU PPL', '知识', 'CMMLU 困惑度评测', 'cmmlu',
           unsupported_reason='当前 Chat Completions 接口不支持 PPL，请选择 Demo CMMLU'),
    _entry('humaneval_cn_gen', 'HumanEval CN', '代码', '中文代码生成能力', 'code',
           languages=['zh'],
           metric_meaning='humaneval_pass@1：每题一个候选在 Docker 中通过真实题目测试的比例，范围 0–100%；当前不提供多候选 pass@10/100。',
           required_local_files=['data/humaneval_cn/human-eval-cn-v2-20210705.jsonl']),
    _entry('gsm8k_gen', 'GSM8K 完整评测', '数学', '完整英文数学文字题，适合稳定评估多步计算', 'gsm8k',
           data_paths=['data/gsm8k/train.jsonl', 'data/gsm8k/test.jsonl'], auto_download=True),
    _entry('math_gen_265cce', 'MATH 完整评测', '数学', '完整竞赛数学，使用经典答案规范化评分', 'math',
           data_paths=['data/math/train.jsonl', 'data/math/test.jsonl'], auto_download=True,
           preparation_note='采用经典 math_gen_265cce，避免当前环境未安装的 math_verify 评分依赖。'),
    _entry('cmmlu_gen', 'CMMLU 完整评测', '知识', '67 个中文学科，逐学科展示知识覆盖', 'cmmlu',
           data_paths=['data/cmmlu/dev', 'data/cmmlu/test'], auto_download=True),
    _entry('mmlu_gen', 'MMLU 完整评测', '知识', '57 个英文学科，评估知识与专业概念辨析', 'mmlu',
           data_paths=['data/mmlu/dev', 'data/mmlu/test'], auto_download=True),
    _entry('bbh_gen', 'BBH 复杂推理', '推理', '23 类困难任务，当前配置拆分为 27 个子任务，覆盖逻辑与状态追踪', 'bbh',
           data_paths=['data/BBH/data'], auto_download=True),
    _entry('IFEval_gen', 'IFEval 指令遵循', '指令遵循', '检查格式、关键词、长度等可验证输出约束', 'ifeval',
           required_modules=['langdetect', 'nltk'], required_nltk_tokenizer=True,
           data_paths=['data/ifeval/input_data.jsonl'], auto_download=True),
    _entry('longbench_multifieldqa_zh_gen', 'LongBench 中文长文问答', '长上下文', '根据长篇中文资料定位信息并回答问题', 'longbench_qa_zh',
           required_modules=['jieba'], data_paths=['data/Longbench'], auto_download=True,
           preparation_note='运行前确认模型配置与服务端允许完整输入长度；256k 上限本身不保证长文回答质量。'),
    _entry('longbench_passage_retrieval_zh_gen', 'LongBench 中文段落检索', '长上下文', '从大量段落中找到目标段落编号', 'longbench_retrieval_zh',
           data_paths=['data/Longbench'], auto_download=True,
           preparation_note='这是模型对已提供文本的定位能力；与外部检索服务及端到端性能压测分别解读。'),
    _entry('truthfulqa_gen', 'TruthfulQA 事实可靠性', '事实可靠性', '检查常见误解和错误前提下的回答', 'truthfulqa',
           unsupported_reason='当前未准备 TruthfulQA 的本地 7B truth judge；此配置需要额外裁判模型和计算资源'),
    _entry('mbpp_gen', 'MBPP Python 编程', '代码', '根据自然语言描述生成代码并用测试验证', 'code',
           source_url='https://github.com/google-research/google-research/tree/master/mbpp',
           required_local_files=['data/mbpp/mbpp.jsonl'],
           unsupported_reason='当前未配置代码评测的隔离执行环境；MBPP 评分会执行模型生成的代码'),
    _entry('mmlu_pro_gen', 'MMLU-Pro 跨学科推理', '综合', '14 个学科的较难多选项知识推理', 'mmlu_pro',
           data_paths=['data/mmlu_pro'], auto_download=True, tags=['热门', '专业知识', '多选项推理']),
    _entry('gpqa_gen', 'GPQA Diamond 科学推理', '科学', '当前配置使用 Diamond 子集，观察高难科学推理', 'gpqa',
           data_paths=['data/gpqa/gpqa_diamond.csv'], auto_download=True,
           tags=['热门', '科学', '高难']),
    _entry('ceval_gen', 'C-Eval 中文学科', '中文', '52 个学科的中文验证集选择题', 'ceval',
           data_paths=['data/ceval/formal_ceval/val', 'data/ceval/formal_ceval/dev'], auto_download=True,
           tags=['中文', '专业知识', '验证集']),
    _entry('math_500_gen', 'MATH-500 数学解题', '数学', '固定 500 道数学题，采用符号验证评分', 'math500',
           required_modules=['math_verify', 'latex2sympy2_extended'],
           data_paths=['data/math/test_prm800k_500.jsonl'], auto_download=True,
           tags=['热门', '数学', '500题']),
    _entry('aime2024_gen', 'AIME 2024 竞赛数学', '数学', '年度高难数学题，适合观察推理上限', 'aime',
           required_modules=['math_verify', 'latex2sympy2_extended'],
           data_paths=['data/aime.jsonl'], auto_download=True, tags=['高难', '竞赛数学', '2024']),
    _entry('IFBench_gen', 'IFBench 复杂指令', '指令', '更多可验证约束类型的指令遵循', 'ifbench',
           required_modules=['emoji', 'syllapy', 'nltk'], required_nltk_tokenizer=True,
           required_nltk_resources=['taggers/averaged_perceptron_tagger_eng', 'corpora/stopwords'],
           required_local_files=['data/IFBench/IFBench_test.jsonl'],
           preparation_note='当前上游数据路径 IFBench 与自动下载映射 IFbench 大小写不一致；请先准备 data/IFBench/IFBench_test.jsonl。',
           tags=['热门', '复杂约束', '需准备数据']),
    _entry('longbenchv2_gen', 'LongBench v2 长文推理', '长文', '真实长资料的较难选择题，按长度与难度看结果', 'longbench_v2',
           data_paths=['data/longbenchv2/data.json'], auto_download=True,
           unsupported_reason='尚未适配按实际 tokenizer 筛选超长样本；完整集含超过当前窗口的题目，不能静默截断后声称全量成绩',
           tags=['热门', '长文推理', '待长度适配']),
    _entry('chinese_simpleqa_gen', 'Chinese SimpleQA 中文事实', '事实', '中文短事实问题，需独立裁判区分正确、错误和未作答', 'simpleqa',
           languages=['zh'], requires_judge=True,
           purpose='检查模型能否简短准确回答中文事实问题，并区分错误回答与未作答。',
           task_example='用中文询问某项公开事件发生的年份，要求给出简短事实答案。',
           suitable_uses=['中文事实问答可靠性', '中文知识助手错误与拒答分析'],
           source_url='https://github.com/OpenStellarTeam/ChineseSimpleQA',
           data_paths=['data/chinese_simpleqa'], tags=['中文', '事实性', '需裁判']),
    _entry('simpleqa_gen', 'SimpleQA 英文事实', '事实', '英文短事实问答，需独立裁判', 'simpleqa',
           requires_judge=True, data_paths=['data/simpleqa/simple_qa_test_set.csv'],
           tags=['英语', '事实性', '需裁判']),
    _entry('humaneval_plus_gen', 'HumanEval+ Python 函数', '代码', '增加测试用例的函数级 Python 代码评测', 'code',
           required_modules=['evalplus'], requires_execution=True,
           source_url='https://github.com/evalplus/evalplus',
           purpose='对生成的 Python 函数执行更充分的测试，检查原有少量测试可能遗漏的错误。',
           metric_meaning='pass@1 等指标表示生成候选通过功能测试的概率或估计值；必须注明 base/plus 测试与采样 k，不能混比。',
           limitations='采用 EvalPlus 扩展测试，不是完整仓库开发任务；需要隔离执行环境、EvalPlus 及匹配数据。',
           required_local_files=['data/humaneval/human-eval-v2-20210705.jsonl'],
           tags=['代码', '扩展测试', '需执行环境']),
    _entry('mbpp_plus_gen', 'MBPP+ Python 小程序', '代码', '自然语言编程要求配合扩展测试', 'code',
           required_modules=['evalplus'], requires_execution=True,
           source_url='https://github.com/evalplus/evalplus',
           purpose='根据简短要求生成 Python 小程序，并用更多测试检查功能与边界条件。',
           metric_meaning='当前 MBPP+ 评分器通过 EvalPlus 汇总 plus 测试通过结果，重点看 mbpp_plus_pass@1，通常为 0–100%。',
           limitations='当前配置的测试集和评分器与普通 MBPP 不同；有限测试通过不等于工程质量，需要隔离执行环境与 EvalPlus。',
           required_local_files=['data/mbpp_plus/mbpp_plus.jsonl'],
           tags=['代码', '扩展测试', '需执行环境']),
    _entry('livecodebench_v6_academic', 'LiveCodeBench v6', '代码', '固定 v6 配置的代码生成、执行和测试输出任务', 'livecodebench',
           required_modules=['pyext'], requires_execution=True,
           tags=['热门', '竞赛编程', '需执行环境']),
    *[_entry(f'ruler_{length}k_gen', f'RULER {length}k', '长文',
             f'{length}k 目标长度下的 13 个合成检索、追踪和聚合子任务', 'ruler',
             required_modules=['tiktoken', 'transformers'], requires_tokenizer=True,
             input_length_target=length * 1024,
             unsupported_reason='未配置实际模型 tokenizer 和输入/输出窗口预算；上游默认 GPT-4 tokenizer，不能据此生成精确的当前模型长度测试',
             preparation_note='每个长度默认 13×100 次请求。需先用实际 tokenizer 核准长度，并为输出和模板留余量；256k 输入不能直接塞满 256k 总窗口。',
             tags=['长上下文', f'{length}k', '需 tokenizer 适配'])
      for length in (32, 128, 256)],
    _entry('bfcl_v4_external', 'BFCL v4 工具调用', '工具', '独立工具调用评测项目，当前尚未适配', 'bfcl',
           integration='external', requires_execution=True,
           unsupported_reason='当前安装没有 BFCL 配置，工具调用、执行环境和专用评分流程均未适配',
           tags=['工具调用', 'Agent', '未适配']),
]


BENCHMARK_PRESETS = [
    dict(id='smoke', label='快速验证',
         description='用少量数学题验证接口与评分流程，不能当作完整能力分。',
         datasets=['demo_gsm8k_chat_gen', 'demo_math_chat_gen']),
    dict(id='daily', label='日常综合',
         description='中文知识、指令、数学与事实可靠性组合；完整题量较大，事实项需先配置独立裁判。',
         datasets=['cmmlu_gen', 'IFEval_gen', 'gsm8k_gen', 'math_500_gen', 'chinese_simpleqa_gen']),
    dict(id='long_context', label='长文与上下文',
         description='区分长文理解与目标长度测试；尚未适配 tokenizer 或超长筛选的项目仍不可运行。',
         datasets=['longbench_multifieldqa_zh_gen', 'longbench_passage_retrieval_zh_gen',
                   'longbenchv2_gen', 'ruler_32k_gen', 'ruler_128k_gen', 'ruler_256k_gen']),
    dict(id='coding', label='编程能力',
         description='基础函数、扩展测试与竞赛代码组合；MBPP 可在已验证的基础环境运行，其他项需各自完成评分适配。',
         datasets=['mbpp_gen', 'humaneval_plus_gen', 'mbpp_plus_gen', 'livecodebench_v6_academic']),
    dict(id='advanced', label='高难推理',
         description='较难跨学科、科学、数学和指令任务，观察上限而非日常能力总评。',
         datasets=['mmlu_pro_gen', 'gpqa_gen', 'aime2024_gen', 'IFBench_gen']),
]


def _normalized(value):
    return re.sub(r'[^a-z0-9]+', ' ', str(value).lower()).strip()


def describe_benchmark(dataset):
    """Describe raw or humanized result abbreviations, with unknowns explicit."""
    normalized = _normalized(dataset)
    selected = next((item for item in sorted(DATASET_CATALOG,
                    key=lambda item: len(item['id']), reverse=True)
                    if normalized == _normalized(item['id']) or
                    normalized.startswith(_normalized(item['id']) + ' ')), None)
    family = selected['family'] if selected else None
    if family is None:
        raw_name = normalized.removeprefix('demo ')
        # Prefixes follow the actual abbreviations emitted by installed configs.
        for prefix, candidate in (
                ('mmlu pro', 'mmlu_pro'), ('gpqa', 'gpqa'), ('ceval', 'ceval'),
                ('math 500', 'math500'), ('aime2024', 'aime'), ('ifbench', 'ifbench'),
                ('longbenchv2', 'longbench_v2'), ('longbench v2', 'longbench_v2'),
                ('chinese simpleqa', 'simpleqa'), ('simpleqa', 'simpleqa'),
                ('ruler', 'ruler'), ('lcb', 'livecodebench'), ('livecodebench', 'livecodebench'),
                ('bfclv4', 'bfcl'), ('bfcl', 'bfcl'),
                ('longbench multifieldqa zh', 'longbench_qa_zh'),
                ('longbench passage retrieval zh', 'longbench_retrieval_zh'),
                ('lukaemon mmlu', 'mmlu'), ('mmlu', 'mmlu'), ('cmmlu', 'cmmlu'),
                ('gsm8k', 'gsm8k'), ('math', 'math'), ('bbh', 'bbh'),
                ('ifeval', 'ifeval'), ('truthful qa', 'truthfulqa'),
                ('truthfulqa', 'truthfulqa'), ('openai humaneval', 'code'),
                ('humaneval', 'code'), ('mbpp', 'code')):
            if raw_name == prefix or raw_name.startswith(prefix + ' '):
                family = candidate
                break
    if family is None:
        explanation = '尚无此评测集的指标解释，请以实际评分配置为准。'
        checks = ['需要先核查错误样本、接口故障和评分规则，不能直接解释为模型能力不足']
        return dict(known=False, label=str(dataset), category='未分类', category_id='unknown',
                    dataset=str(dataset),
                    languages=[], scope='full', tags=[], requires_judge=False,
                    requires_execution=False, integration='unknown',
                    purpose='此自定义或未识别评测的用途尚未登记，需要先查阅任务定义。',
                    task_example='尚无已核实的题型示例。',
                    suitable_uses=['核实任务定义与评分规则后再选择用途'],
                    focuses=['需查阅该评测集的任务与评分器说明'],
                    metric_meaning=explanation, metric_explanation=explanation,
                    low_score_reasons=checks, low_score_checks=checks,
                    impacts=['尚不能从未知指标推断具体业务影响'],
                    limitations='该结果未匹配到目录中的评测定义。', source_url='',
                    interpretation_note=INTERPRETATION_NOTE)
    result = deepcopy(_PROFILES[family])
    purpose, example, uses = _PURPOSES[family]
    result.update(purpose=purpose, task_example=example, suitable_uses=list(uses))
    result.update(known=True, family=family, dataset=str(dataset),
                  interpretation_note=INTERPRETATION_NOTE)
    category_id = _FAMILY_CATEGORIES[family]
    result.update(category_id=category_id, category=_CATEGORY_LABELS[category_id],
                  languages=list(_FAMILY_LANGUAGES.get(family, ['en'])),
                  scope='demo' if normalized.startswith('demo ') else 'full',
                  tags=[_CATEGORY_LABELS[category_id]],
                  requires_judge=family in {'truthfulqa', 'simpleqa'},
                  requires_execution=family in {'code', 'livecodebench', 'bfcl'},
                  integration='external' if family == 'bfcl' else 'opencompass')
    # Raw old abbreviations do not identify a config revision, but these
    # distinctive names still identify the correct task and language variant.
    inferred = None
    if selected is None:
        for prefix, config_id in (
                ('chinese simpleqa', 'chinese_simpleqa_gen'),
                ('openai humaneval cn', 'humaneval_cn_gen'),
                ('humaneval plus', 'humaneval_plus_gen'),
                ('mbpp plus', 'mbpp_plus_gen')):
            if normalized == prefix or normalized.startswith(prefix + ' '):
                inferred = next(item for item in DATASET_CATALOG if item['id'] == config_id)
                break
    metadata = selected or inferred
    if metadata:
        for key in ('label', 'category', 'focuses', 'metric_meaning',
                    'low_score_reasons', 'impacts', 'limitations', 'source_url',
                    'purpose', 'task_example', 'suitable_uses', 'category_id',
                    'languages', 'scope', 'tags', 'requires_judge', 'requires_execution', 'integration'):
            result[key] = deepcopy(metadata[key])
    if selected:
        result['config_id'] = selected['id']
    # Even unprefixed old Demo files need the small-sample qualification.
    if 'demo' in normalized.split() and not selected:
        result['limitations'] = 'Demo 样本少，只适合接口验证，不宜与完整榜单比较。' + result['limitations']
        result['purpose'] = '用少量样本验证接口和评分流程；' + result['purpose']
        result['suitable_uses'] = ['首次验证模型接口', '调整提示词后的快速回归']
    if family == 'code' and normalized.startswith('mbpp') and metadata is None:
        result['label'] = 'MBPP'
        result['source_url'] = 'https://github.com/google-research/google-research/tree/master/mbpp'
    if family in ('cmmlu', 'mmlu', 'ceval', 'mmlu_pro'):
        marker = {'cmmlu': 'cmmlu ', 'mmlu': 'lukaemon mmlu ',
                  'ceval': 'ceval ', 'mmlu_pro': 'mmlu pro '}[family]
        subject = normalized.rsplit(marker, 1)[-1] if marker in normalized else ''
        if subject and subject.split()[0] not in {'gen', 'chat', 'base'}:
            result['subject'] = subject
            subject_label = _SUBJECT_LABELS.get(subject.replace(' ', '_'), subject)
            result['subject_label'] = subject_label
            result['label'] = result['label'].removesuffix(' 完整评测') + ' · ' + subject_label
            language = '中文' if family in {'cmmlu', 'ceval'} else '英文'
            result['purpose'] = f'检查模型能否用{language}理解“{subject_label}”选择题，并辨别相关概念和知识。'
            choices = '若干候选选项' if family == 'mmlu_pro' else 'A/B/C/D'
            result['task_example'] = f'阅读一道{subject_label}题，从{choices}中选择最符合概念或事实的答案。'
            if 'demo' in normalized.split():
                result['purpose'] = '用少量学科题验证接口和评分流程；' + result['purpose']
            if selected and selected['id'].endswith('_ppl'):
                result['purpose'] += '采用选项概率比较，当前聊天接口不可运行。'
            result['suitable_uses'] = [f'{subject_label}资料理解', f'{language}学科知识问答']
            for terms, direction, impacts in _SUBJECT_DOMAINS:
                if any(re.search(r'\b' + re.escape(term) + r'\b', subject) for term in terms):
                    result['focuses'].insert(0, direction + '（当前子任务）')
                    result['impacts'] = list(impacts)
                    result['low_score_reasons'].insert(0, f'可能{subject_label}相关知识或概念辨析不足')
                    break
    result['metric_explanation'] = result['metric_meaning']
    result['low_score_checks'] = deepcopy(result['low_score_reasons'])
    return result


def _data_locations(relative, root):
    cache_root = Path(os.environ.get('COMPASS_DATA_CACHE') or root)
    return (cache_root / relative, Path.home() / '.cache/opencompass' / relative)


def dataset_preparation(entry, root):
    """Inspect readiness without loading or downloading any dataset."""
    paths = entry.get('data_paths') or entry.get('required_local_files') or []
    cached = all(any(path.exists() for path in _data_locations(relative, root))
                 for relative in paths)
    result = dict(data_cached=cached if paths else None,
                  preparation_note=entry.get('preparation_note', ''))
    if paths and not cached and entry.get('auto_download'):
        extra = '数据尚未缓存；首次开始评测时 OpenCompass 会下载数据，需要可访问数据源。'
        result['preparation_note'] = (result['preparation_note'] + ' ' + extra).strip()
    return result


def unavailable_reason(entry, root):
    """Return a concrete runtime prerequisite failure, never config existence."""
    if entry.get('environment_installable'):
        from code_runtime import code_readiness
        return code_readiness(root, entry['id'])
    if entry.get('unsupported_reason'):
        return entry['unsupported_reason']
    for module in entry.get('required_modules', []):
        if importlib.util.find_spec(module) is None:
            return f'未安装 {entry["label"]} 评分依赖 {module}'
    if entry.get('required_nltk_tokenizer'):
        try:
            import nltk
            nltk.data.load('nltk:tokenizers/punkt/english.pickle')
        except (ImportError, LookupError):
            return f'未准备 {entry["label"]} 句子计数所需的 NLTK punkt/punkt_tab 英文资源'
    if entry.get('required_nltk_resources'):
        try:
            import nltk
            for resource in entry['required_nltk_resources']:
                nltk.data.find(resource)
        except ImportError:
            return f'未安装 {entry["label"]} 评分依赖 nltk'
        except LookupError:
            return f'未准备 {entry["label"]} 评分资源：{resource}'
    for relative in entry.get('required_local_files', []):
        if not any(path.is_file() for path in _data_locations(relative, root)):
            return f'未准备 {entry["label"]} 数据：{relative}'
    return ''
