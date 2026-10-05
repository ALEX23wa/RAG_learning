# Quality-v2 评测脚本使用说明

这些脚本收集可追溯的实验数据，不会自动批准上线。题集源审核、问答独立审核、运行性能和发布校验分别保留，缺项不补造为通过。

实验已结束，完整成绩与是否达标见 [结项报告](quality-v2-status.md)。以下保留复现方法；既有报告、题集与索引不得原地覆盖。

## 输入与版本冻结

原v1私有题集位于 `.rag/evaluation/quality-v2/`，封存的等价证据副本位于 `.rag/evaluation/quality-v2-evidence-v2/`；两版开发集和保留集均各120题，总计覆盖38份原资料，跨来源、多轮、不可回答各24题。新实验显式选择evidence-v2，历史v1报告继续保留。`agent_source_reviewed` 表示智能体独立检查过原资料，不能写成 `human_verified`。纯图题不计入自动文本 span 召回。

每次运行绑定题集字节 SHA256、相邻 `manifest.json`、`source-checks.json`、运行配置，以及代码、索引和资料运行清单。已经存在结果却没有运行清单的旧报告不能续跑。代码、题集、模型、索引或配置改变时，使用新的报告路径；不要删除原报告的指纹字段强行恢复。

运行清单同时保存两种资料状态：`documents` 是已入库版本，`observed_sources` 是磁盘原文件的实际哈希与可读状态，`source_inventory` 是当前支持格式的文件清单。未同步的修改、删除或新增文件会改变运行指纹。已有可用文档的磁盘哈希与入库哈希不一致时，评测会拒绝开始或继续。受限 API 的 `source_fingerprints` 用路径摘要代替目录名，但也反映实际哈希失配。

原文件哈希在进程内按路径、文件大小、纳秒修改/变更时间、设备号及 inode 缓存。这是本地 stat 缓存优化，不是防篡改保证；正式发布仍独立完整读取并校验原文件。清单观察只读目录和文件，不会触发 OCR、解析、同步或付费调用。

## 候选索引配置

以下 PowerShell 命令均从仓库根目录运行：

```powershell
$python = 'C:\Users\23245\.conda\envs\langchain\python.exe'
$settings = '.rag/reports/quality-v2/candidate-input-v5-settings.json'
$cases = '.rag/evaluation/quality-v2-evidence-v2/dev.json'
```

候选settings必须选择已经完成构建的代次，将该代次的 `collection`、`catalog` 和 `settings` 合并成一个平面 `Settings` 对象；上例对应已构建的v5 E5候选，并不代表三模型对照完成或胜出。直接填写代次文件路径或只填写 `config/quality-v2.json`，不会自动切换到候选库。当前full-v5推理窗口保持串行，不与本文示例重叠启动；具体生成步骤见 [实施说明](RAG质量优化实施.md)。

两个离线入口的 `--settings` 都只接受已知 `Settings` 字段；`root`、`state`、`corpus` 转换为路径，**相对路径以 JSON 所在目录为基准**。`load_active` 被强制设为 `false`，避免现有 `active.json` 覆盖指定的候选代次。可显式填写绝对路径以减少歧义。配置只存模型名称、目录及环境变量约定，不存 API 密钥。

## 纯检索消融

原代码基线另用 `scripts/compare_retrieval.py --variant baseline`。在首次运行前，先补录其实际资产：

```powershell
& $python scripts/freeze_baseline_assets.py freeze --batch-size 16
& $python scripts/freeze_baseline_assets.py verify --batch-size 16
```

补录文件为 `.rag/experiments/quality-v2-baseline/asset-manifest.json`。它明确记录本次检查时间，以及最初快照**未记录资产哈希**的事实；当前哈希不能证明过去的模型或向量资产从未改变。补录涵盖 SQLite 所有逻辑表（包括 chunks、FTS 及其索引表）、实际模型文件，以及 Qdrant `papers_e5_v1` 的每个 point ID、向量、payload 和集合配置。文件哈希流式读取，Qdrant 按 ID 顺序分批滚动读取，不加载模型也不写集合。

基线脚本在运行前后完整复核这些资产，将资产清单哈希绑定到报告；已有 SQLite 工作副本也必须内容一致。旧 `Store` 初始化产生的额外空表允许存在，已冻结表的内容/结构变化或额外非空表会被拒绝。前后校验不一致的报告标记 `asset_verification_failed`，禁止拼接续跑；进程中途退出、已有结果却没有完成后校验时，也必须改用新的报告文件。正常的 `--max-queries` 分批运行会完成后校验，可继续剩余题目。

原实现仍以 `legacy_unversioned` 标记索引配置，候选池没有暴露的 Recall@80 继续为未知，不从最终十条结果反推。

```powershell
& $python -m ragapp.experiments --mode offline --settings $settings `
  --cases $cases --report .rag/reports/quality-v2/evidence-v2-v5-retrieval-rrf30.json `
  --config hybrid_rerank --rrf-k 30 --expand-context --max-queries 10
```

移除 `--max-queries 10` 并重跑同一命令即可继续剩余题目；这个参数限制的是本次新增题数。每个消融配置应使用单独的报告文件。

| 参数 | 含义 |
| --- | --- |
| `--config bm25/dense/hybrid/hybrid_rerank` | 分别测关键词、向量、融合、融合加重排 |
| `--rrf-k` | 覆盖融合常数；正整数，开发阶段可比较 10、30、60 |
| `--expand-context` / `--no-expand-context` | 开启或关闭上下文扩展 |
| `--max-queries` | 限制本次新增检索题数 |
| `--allow-blind` | 配置冻结后显式允许保留集评测 |

默认消融关闭查询规划，只运行本地检索和模型，不调用在线生成 API。离线模式会加载配置的本地 embedding/reranker，可能启动或连接 Qdrant。

HTTP 模式使用已有服务，通过环境变量提供有 `/api/search` 调试权限的令牌，令牌不写入命令参数：

```powershell
# RAG_EVAL_TOKEN 事先由本机授权配置设置为 admin 令牌。
& $python -m ragapp.experiments --mode http --cases $cases `
  --report .rag/reports/quality-v2/evidence-v2-http-hybrid.json --config hybrid `
  --token-env RAG_EVAL_TOKEN --url http://127.0.0.1:8000
```

HTTP 消融不能用 `--settings` 切换服务索引或模型。模型对照需启动配置独立的服务。消融报告的 RSS 只覆盖运行脚本所在进程，HTTP 下也不含服务器；它不满足服务加数据库的发布性能门槛。

检索报告提供真实 span Recall@80、Recall@10、去重 nDCG@10、上下文覆盖、micro/query macro/document macro、按证据簇 bootstrap 的 95% 区间及延迟分位数。`legacy_locator_recall_at_10` 只说明物理定位命中，不能替代真实 span 指标。候选和上下文指标必须有实际 trace；缺少 trace 时为 `null`，不是零也不是通过。消融延迟分位数使用插值，正式答题性能验收则使用下文的 nearest-rank p95。

严格 span 匹配只进行 NFC 和空白归一化，保留大小写、数学符号、上下标和标点。目标文本可以跨越同一可见证据项内的多个 `source_spans`，但必须逐字符核对物理来源及明确的相邻关系；同页、同函数或未展开的父段本身不能算命中。跨页误归属、未映射的正文、未知来源缺口及重叠映射都会被拒绝。

`scripts/audit_corpus_coverage.py` 是开发集全库覆盖诊断，不执行检索、不读取盲测题。它另外报告 PDF 排版形式敏感性：`ragapp.evidence_text` 只转换显式印刷标点，保留表格单元格边界、科学符号和原字符映射。这项诊断不代替严格 Recall，不参与发布门槛；普通文本与版面解析的表示差异也不能自动认定为内容丢失。

`corpus-coverage-v2` 的 `summary`、`by_source`、`by_type` 都以必须证据组为单位：同一组多个等价表示取 OR，只计一次；106 道可文本评分开发题的分母仍是 118 组。`alternative_summary` 与带有 `alternative_index` 的 `evidence_diagnostics` 保留逐表示诊断，不能将其分母代替证据组。每个来源/类型只用本分组内的表示判断覆盖；若一个证据组允许不同来源，则它可能出现在多个来源桶，桶总数不保证可相加。`case_coverage` 继续要求该题全部必须证据组都被覆盖。

严格覆盖调用当前评分器，可接受同一文本中有明确原文连续关系的多个物理 `source_spans`；父节采用完整的 `parent_source_spans` 映射。`mapped_child_or_parent` 另作宽松诊断，只说明相邻显示片段可拼出文本，未证明原文相邻关系，不能替代严格覆盖。跨检索块的拼接也只供定位分块问题，不表示一次检索已取到可引用的完整上下文。

## 离线真实问答

```powershell
& $python scripts/run_quality_answers.py --settings $settings --cases $cases `
  --report .rag/reports/quality-v2/evidence-v2-dev-answers.json `
  --profile production --stage development --max-queries 5
```

该入口调用真实 `Service.ask`，会按 profile 使用在线生成、核验和必要视觉调用；受项目既有额度和 campaign 台账约束。默认题集是开发集，默认 profile 为 `production`，默认 stage 为 `development`。`eval_aihub_free` 是显式可选 profile，脚本不会因免费额度不足自动换成付费通道。

同一命令去掉题数限制后可断点续跑；已完成题目不会重问。失败题保留 `failed_attempts`，再次运行会尝试这道尚未完成的题，因此历史失败费用仍计入报告。profile、stage 或脚本内容变化也要求新建报告。恢复时检查结果 ID 唯一且属于该题集，并要求已完成题目的资源记录完整对应。

保留集只在配置已经冻结后执行，例如：

```powershell
& $python scripts/run_quality_answers.py --settings $settings `
  --cases .rag/evaluation/quality-v2-evidence-v2/blind.json `
  --report .rag/reports/quality-v2/evidence-v2-blind-answers.json `
  --profile production --stage acceptance --allow-blind
```

每题输出包括原始答案、检索轨迹、候选与实际上下文、辅助自动评分、耗时、费用调用记录。自动准确率只是待独立核对的参考值；收齐题目后的状态为 `awaiting_manual_audit`，不会直接变成 `passed`。

辅助数字评分会剔除 `[S42]` 等引用标号，要求完整数字匹配，并核对金标声明的单位和条件。没有 `numeric` 约束时返回 `not_evaluated`，不能报告为数字正确率；旧题只有数值和容差时明确标记为数值检查。主张核验模型列出的主张不一定覆盖整段答案，正式事实支持率仍需独立逐主张审核。

费用从实际 `service.models.store` 读取；启用 campaign 时这是共享的 `billing.sqlite3`，不是文档索引库。成功、失败和未结算预留均保留。逐题费用使用问答调用期间新增的账目，因而精确逐题归属需要此时没有其他并行付费请求；并发请求可能使此上界偏高。`campaign_cost` 则扫描本地各代次账本，按 call ID 去重并保留较大结算额/预留额，避免漏掉复制账本或失败费用。

## 真实资源记录

每次启动或恢复先执行固定的纯检索预热，关闭在线查询规划；预热不使用题集问题、不计入答题延迟和 RSS 样本。计时范围是预热后的 `Service.ask`，包括本题检索、在线请求和核验，不包括报告序列化与外部清单检查。

RSS 来自 `psutil` 实际观察的两个不同本机进程：当前问答 worker，以及配置端口上监听且进程名为 Qdrant 的数据库进程。远端数据库、无法确定监听进程、进程读取失败，都不能用脚本自身 RSS 顶替。

`performance_measurement` 的关键字段如下：

```json
{
  "scope": "offline_service_and_database",
  "measurement_method": "psutil_process_rss",
  "warmup_excluded": true,
  "service_pid": 1234,
  "database_pid": 5678,
  "samples": 120,
  "process_sessions": [
    {"service_pid": 1234, "database_pid": 5678, "warmup_excluded": true}
  ],
  "records": [
    {
      "id": "示例题号",
      "elapsed": 4.2,
      "service_pid": 1234,
      "database_pid": 5678,
      "service_rss_bytes": 1000000000,
      "database_rss_bytes": 300000000
    }
  ]
}
```

上面仅展示结构，不是实测结果。实际 `samples` 和 `records` 数量必须与已完成题目一致，每题唯一。恢复后新的 worker PID 登记为另一个 session，并写入相应题目的记录。

采样间隔为 100 ms，并保留开始及结束样本。峰值是**同一次采样的服务 RSS 加数据库 RSS**的最大值，不将各进程在不同时刻的峰值相加。所有原始成对样本保存在 `rss_samples`。采样可能漏过短于采样间隔的瞬时峰值，因此记录同时保留 `sampling_interval_seconds`。

正式 p95 按 `sorted(elapsed)[ceil(0.95*n)-1]` 重算，峰值也从逐题原始记录重算；不信任手工填写的汇总标量。HTTP 客户端内存、缺少数据库 RSS、伪造或未登记的 PID、无效数值、遗漏题目的测量均不能通过资源门槛。

## 独立审核和发布

`ragapp.audit.finalize` 合并与 dataset/runtime 指纹绑定的逐题独立审核。每题需源文件核对、金标核对、理由、通过与引用判定，以及支持主张数/总主张数。它按 quality-v2 的 90% 准确率、90% span Recall@10、95% 忠实度、95% Recall@80、拒答 precision/recall 各 90%、引用位置有效率 100%、p95 不超过 60 秒、合计 RSS 不超过 12 GiB 判定；缺指标保留 `awaiting_measurements`。

完整发布另需冻结保留集 120 题、审核者独立性声明、同版本旧 60 题回归、campaign 总费用等材料。参考 `ragapp.maintenance.publish_generation` 的完整报告契约。脚本只产出已实测部分，不会伪造审核者、旧回归结果或发布通过状态。

本轮 `campaign=quality-v2` 的新增实验费用累计上限为100元（准备20/开发50/验收30），包括已有调用及失败预留，续跑不重置。只有实测及独立审核全部通过后才手动发布；候选构建、单元测试与重评分都不会切换生产，旧索引及回滚资产继续保留。

## 无模型测试

```powershell
& $python -m pytest tests/test_quality_evaluation.py tests/test_evaluation.py `
  tests/test_publish_generation.py tests/test_index_fingerprint.py -q
```

这些测试使用模拟问答与进程读数，并检查实际临时 SQLite 台账；不调用生成 API，也不加载 embedding/reranker。私有题集存在时只核对题数、哈希和隔离约束，不改动题目。
