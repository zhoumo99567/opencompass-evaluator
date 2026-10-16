# OpenCompass LAN Evaluator

一个面向局域网本地大模型的 OpenCompass 评测与报告查看器。它使用 OpenAI-compatible API 连接本地模型，并提供一个本地网页来选择评测集、保存评测集合、启动评测、查看运行结果和生成历史报告。

## 快速开始

需要 Python 3.10+、已安装 `uv`，以及一个兼容 OpenAI Chat Completions 的模型接口。

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python opencompass -r requirements-viewer.txt
cp .env.example .env
```

编辑 `.env`：

```dotenv
OPENCOMPASS_BASE_URL=http://127.0.0.1:8080/v1/chat/completions
OPENCOMPASS_MODEL=your-model-name
OPENAI_API_KEY=your-api-key
OPENCOMPASS_QPS=1
```

启动网页：

```bash
./run_viewer.sh --host 127.0.0.1 --port 8765
```

打开 <http://127.0.0.1:8765/>，首页直接进入“开始评测”。主导航提供“开始评测”“性能测试”“历史报告”“模型设置”四个入口。勾选评测集并启动 OpenCompass 后才显示运行进度；已有评测结果可从侧栏“评测记录”打开。切换页面不会中断进度更新。评测结果会写入 `outputs/`，报告会写入 `reports/`；这些目录默认不会提交到 Git。

也可以直接运行 OpenCompass：

```bash
./run_opencompass.sh --models ./lan_eval.py --datasets demo_gsm8k_chat_gen
```

## 安全说明

`.env`、网页保存的设置、评测输出、报告和运行时缓存都已加入 `.gitignore`。提交代码前请确认不要把 API key 写入 Python 文件、日志或报告。

## 当前网页能力

- 选择一个或多个 Demo GSM8K、MATH、CMMLU、HumanEval CN 等评测集
- 保存、查看和应用自定义评测集合
- 从网页启动 OpenCompass 并查看运行状态与日志路径
- 浏览分数、预测样本和运行文件
- 调用已配置的本地模型生成中文 Markdown 评测报告

## 多选评测与可用性

网页按所选评测集生成完整配置。Chat/Base 版本使用独立结果名称，避免文件覆盖。
CMMLU 会展开成多个学科子任务；页面会显示预期与已完成的数量，结果未齐不会标为完成。

- 当前 OpenAI Chat Completions 适配器不支持 PPL，页面禁用该选项。
- HumanEval CN 需要将数据放在 `data/humaneval_cn/human-eval-cn-v2-20210705.jsonl`
  （项目目录或 OpenCompass 数据缓存目录下），并安装 OpenCompass 兼容的 `human_eval` 评分依赖；缺失时页面禁用该选项。
- 较早版本始终只运行 `lan_eval.py` 中的 GSM8K；旧结果不会自动补齐，请重新选择并启动评测。

验证多选配置与部分结果状态：

```bash
.venv/bin/python -m unittest discover -s tests -v
```

## 鹈鹕骑自行车 SVG 测试

在“开始评测”中切换到“鹈鹕 SVG 测试”，点击“生成动画 SVG”。
工具复用模型设置，单轮发送动画版提示词，要求鹈鹕骑自行车、车轮旋转、腿部蹬踏及翅膀轻摆，使用内嵌 CSS 关键帧，不使用脚本或外部资源；每次报告保留实际提示词。
使用 temperature=0.2，默认 max_tokens=32768，可在页面选择 8k–256k 的生成额度（包含思考 tokens）。采用流式接收，显示思考/正文字符数，并保存已经接收的正文；可以停止生成。客户端总时限为 10 分钟，模型服务或网关可能更早终止。256k 上下文不等于可生成 256k，实际可用生成额度取决于模型服务配置。生成时可以切换页面，结果自动保存到 `reports/`。
若思考耗尽额度而没有最终回答，会保留结束原因和 token 用量，并提示调整额度。

HTTP 504 表示网关或上游等待超时，增加客户端等待时间或生成额度并不会自动改变网关时限。流式响应可减少等待完整生成造成的空闲超时，但无法保证绕过网关的总请求时限。需结合服务端日志核对超时、缓冲和思考预算；不会自动重复昂贵请求。

页面可选择“请求关闭思考（Qwen 兼容）”，发送 `chat_template_kwargs.enable_thinking=false`，参数和实际思考字符数随报告保存。该参数是 [Qwen/vLLM 扩展](https://qwen.readthedocs.io/en/latest/deployment/vllm.html#thinking-non-thinking-modes)，接口可能不支持或忽略；默认保持模型服务设置。若目的是快速验证动画 SVG，可先用 8k/16k 输出额度与关闭思考选项验证，而不是持续增加到 128k。
重新打开网页时，同一模型会沿用最近一次鹈鹕测试的生成额度与思考模式，可在页面调整。

- 保留 CSS 关键帧动画（如车轮旋转、翅膀摆动）及减少动态效果设置；历史报告会从原始输出重新构建预览。模型未生成动画时仍显示静态图。
- 大幅 SVG 预览、耗时、接口返回的 token 用量、提示词与原始输出。
- 历史报告中的图片画廊，任选两次成功结果并排比较。
- 更换模型设置后再次生成，可与旧模型结果对照。
- 预览移除脚本、外部资源等活动内容；原始输出以文本保存。可从预览区或历史画廊下载包含动画样式的独立 SVG，用浏览器打开即可播放。静态旧报告仍按静态预览展示。
- 超时、无效 SVG、服务重启导致的中断会显示为失败，不伪造结果。

该测试源自 [Simon Willison 的实验](https://simonwillison.net/2024/Oct/25/pelicans-on-a-bicycle/)，
用于非正式观察绘图和指令遵循，不提供客观准确率，不参与 OpenCompass 标准分数。
固定提示词和参数方便比较，但不同模型的生成仍可能具有随机性。

## 展示参考

- [EvalScope](https://github.com/modelscope/evalscope)：模型/数据集总览、对比和报告组织。
- [Promptfoo](https://www.promptfoo.dev/docs/usage/web-ui/)：并排输出比较、人工审阅与失败筛选。
- [Inspect View](https://inspect.aisi.org.uk/log-viewer.html)：逐题消息、评分依据和实时状态。

本项目没有复制这些项目的代码或引入其前端依赖。

## 数据集解读与端到端性能测试

左侧“性能测试”可以设置并发与目标输入长度矩阵，记录首生成响应、首最终文本、完整响应分位数、估算 TPOT、整体吞吐、失败率和逐请求数据。可设置业务 SLO、停止测试，并在历史中下载 JSON/CSV。输入长度为估算，实测 token 长度以接口 usage 为准。

选择卡片和每条能力结果默认显示“测什么”“适合用途”“题型示例（非原题）”。评分含义、低分排查候选、解读局限和官方来源紧邻该项展开查看；CMMLU 等分学科结果显示中文学科名，并保留原始配置 ID。结果支持按名称、学科或用途搜索并分页。完整集与 Demo 分开说明，缺评分依赖的条目会显示具体准备条件。

目录按能力分类，覆盖中文知识、综合知识、高难科学、数学、逻辑、指令遵循、编程、长上下文、事实可靠性和工具调用。能力页与结果页均支持分类筛选；能力页还可按语言、Demo/完整配置、运行条件和用途搜索，默认每页 8 项。切换筛选和翻页不会丢失勾选，已选区可逐项移除或清空。

预设提供“快速验证”“日常综合”“长文与上下文”“编程能力”“高难推理”。点击会替换当前选择，只包含可运行项；暂不可运行项会列出原因，全部未就绪的预设不能启动。预设名称不代表低耗时或官方综合榜单，正式完整集的题量可能较大。

新增 MMLU-Pro、GPQA Diamond、C-Eval、MATH-500、AIME 2024，以及 IFBench、LongBench v2、SimpleQA、EvalPlus、LiveCodeBench、RULER 和 BFCL 的用途与准备条件。当前目录 33 项；配置、评分依赖和运行环境准备完成后才可选，不会把“已列入目录”当作“可以直接运行”。部分数据首次执行会自动下载，规则评分资源需准备一次：

```bash
uv pip install --python .venv/bin/python -r requirements-viewer.txt
.venv/bin/python -m nltk.downloader punkt punkt_tab averaged_perceptron_tagger_eng stopwords
```

IFBench 当前上游自动下载映射存在路径大小写不一致，需先把数据准备到 `data/IFBench/IFBench_test.jsonl`。裁判类评测仍需独立裁判配置，代码类需要执行环境；RULER 需实际模型 tokenizer，LongBench v2 需超长样本处理，BFCL 需独立适配。这些条件未满足时网页保留说明并禁止勾选。

### 网页准备代码评测环境

在本机打开网页左侧“环境管理”，点击“安装代码评测环境”。后台会启动已安装的 OrbStack/Docker Desktop，构建固定的 Python 评分容器，验证正确答案、错误答案与超时处理，并下载、校验 MBPP 官方数据。进度和日志持续保存在本地；失败时可以重试，刷新页面也能继续查看。运行评测时暂不能安装环境，安装期间也不能启动新评测。

安装发生在运行网页后台的电脑，模型继续使用已保存的局域网接口。电脑需要先安装 Docker/OrbStack；网页可以启动已有的容器应用，不能完成操作系统安装向导。安装按钮仅接受本机页面请求。

基础适配支持 **MBPP** 和 **HumanEval CN**，各项数据条件独立检查。MBPP 可以自动准备；HumanEval CN 尚无已核验的中文自动下载源，需把包含真实中文 `prompt`、`task_id`、`test`、`entry_point` 的 164 题 JSONL 放到 `data/humaneval_cn/human-eval-cn-v2-20210705.jsonl`，然后重新检测。不会用英文 HumanEval 冒充中文，也不会因为缺少 CN 数据而阻止已经就绪的 MBPP。

模型生成的代码只在无网络、只读根目录、非 root 用户且限制 CPU、内存、进程数和执行时间的容器中评分。当前每题生成一个候选，报告为 pass@1/通过率。HumanEval+、MBPP+、LiveCodeBench 和 BFCL 尚需独立评分或工具适配，安装基础环境后仍不可直接运行。

环境状态保存在已忽略的 `.cache/` 中，数据保存在已忽略的 `data/` 中。更新评分器或镜像后需要重新安装/自检；服务重启导致的未完成任务会标为中断，可重新检测或修复。

性能测试默认生成预算为 256 tokens，用于短输出吞吐验证。思考型模型可能在这个预算内只返回思考，页面会分别记录生成成功与最终文本成功；测完整回答延迟时应增加预算，例如 2048 或 4096 tokens，并检查截断次数。

指标定义、低分解读与主流开源方案见 [BENCHMARK_GUIDE.md](BENCHMARK_GUIDE.md)。

## 命令行调用

可以直接在终端启动评测、查看数据集用途、读取分数和保存 JSON，无需打开浏览器。`run_cli.sh` 使用项目的 uv 虚拟环境。它通过后台 HTTP API 复用网页的模型设置、多选评测配置和结果解析；网页保存的设置优先于 `.env`，不需要在命令里填写密钥。

先确保后台已启动；已经运行时无需重复启动：

```bash
./run_viewer.sh --host 127.0.0.1 --port 8766
```

在另一个终端调用：

```bash
# 帮助、评测集列表；指定 ID 查看用途、题型示例、评分及低分排查
./run_cli.sh --help
./run_cli.sh datasets
./run_cli.sh datasets demo_gsm8k_chat_gen
./run_cli.sh datasets --category math --scope full --available
./run_cli.sh datasets --search 科学 --json

# 一次选择多个评测集，等待结束并保存完整 JSON
./run_cli.sh ability run \
  --datasets demo_gsm8k_chat_gen demo_math_chat_gen \
  --max-seq-len 262144 --max-out-len 32768 \
  --wait --wait-timeout 7200 --json --output result.json

# 查询当前任务、历史原始结果、最新已完成结果
./run_cli.sh ability status --json
./run_cli.sh runs --json
./run_cli.sh results latest --json
./run_cli.sh results 你的运行ID --json --output result.json

# 查看保存的报告，与原始能力结果分开查询
./run_cli.sh reports --kind analysis --json
./run_cli.sh reports 你的报告ID --json
```

`results` 返回原始 `scores`、`score_interpretations`（每个数据集的用途、指标与影响）和最多 250 条预测样本预览；全量预测保存在该次运行的 `outputs/**/predictions/`。`latest` 只选择结果完整的记录。`ability run --wait` 通过后台返回的 `run_ids` 获取本次任务的结果，避免混入其他评测。省略 `--wait` 时立即返回任务信息，任务在后台继续运行。

也可以调用鹈鹕与端到端性能测试，结果通过 `reports` 查询：

```bash
./run_cli.sh pelican run --max-tokens 8192 --thinking-mode disabled --wait --json
./run_cli.sh performance run --concurrency 1,2,4 --input-tokens 1024,4096 \
  --requests-per-cell 8 --output-tokens 256 --wait --json
./run_cli.sh performance status --json
./run_cli.sh pelican stop
./run_cli.sh performance stop
```

后台使用其他端口时，将全局参数放在命令前，例如 `./run_cli.sh --server http://127.0.0.1:8765 datasets`；也可设置 `OPENCOMPASS_VIEWER_URL`。`--json` 保证 stdout 只有最终 JSON，等待进度写入 stderr，适合脚本和 CI。`--output` 保存的文件权限为 600。

退出码：成功或任务已启动为 `0`，终态失败/停止或结果不完整为 `1`，参数/连接错误为 `2`，等待超时为 `124`，Ctrl+C 为 `130`。等待超时或 Ctrl+C 只结束客户端等待，不自动停止后台任务；可以继续查询，或显式调用相应的 `stop`。

若希望完全不启动网页后台，可继续使用原生 `./run_opencompass.sh --models ./lan_eval.py --datasets demo_gsm8k_chat_gen`。该脚本直接运行 OpenCompass，使用 `.env`，不读取网页另存的模型设置；输出仍写入本地 `outputs/`，原生 CLI 参数与本节的 API 客户端参数不同。
