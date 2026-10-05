# RAG 质量优化实施与复现实验

本轮优化已完成实现与实验并按用户要求收尾。最佳完整开发轮次为 dev-v6（严格来源审核117/120），对应可移植参数在 `config/quality-v2-best.json`；新盲测完整完成，但召回、拒答和延迟未达到原发布门槛，未切换生产索引。最终成绩及限制见 [结项报告](quality-v2-status.md)。

## 已实现的处理链

- `/api/search` 与 `/api/chat` 共用查询理解。普通主题词只提供软召回偏好；明确文件名、路径或 `document_ids` 才限制来源。重写和补检继承原问题的明确范围。
- 原问题始终参与召回；指代或比较问题最多调用一次规划，最多增加三个查询。历史只用于消解指代。失败保留当前问题及已发生调用的费用。
- BM25、向量及软定位候选统一 RRF；不再在融合后将 focused 候选置顶。向量候选有界补取并剔除旧版本，查询和重排正文按 tokenizer 窗口处理，避免尾部静默丢失。
- 正文采用 384 token、48 token 重叠；合并同节短段，保留原文空白、字符范围和物理定位。Python 标注 AST 函数/类/导入并优先按语句边界切分；单条超长语句仍需 token 窗口。其他语言标记 `text_fallback`，不推断调用图。
- PDF 支持 auto/text/layout 和物理页覆盖；auto 检查文字质量、列布局信号。MinerU 缺页或空结果显式失败。缓存绑定源文件哈希、解析路径和结果校验和。
- 同一 PDF 页、同一章节的标题和正文共享父节；表格、图片和公式保持边界。v4 使用 `structure-v4-pdf-section-parents`；当前v5指纹为 `structure-v5-full-embedding-input-budget`，新增完整embedding模板与special tokens的预算核对。解析指纹独立，因此可复用校验过的 MinerU 解析缓存。
- 表格建立表头说明和带表头的行组；确定性计算器只接受固定操作，不执行生成代码。图片描述帮助召回，问答时针对原图/物理 PDF 页视觉复核。
- 上下文组装按文档覆盖、父节扩展、重复父块消除及预算工作；使用紧凑物理来源元数据，生成前重新核对预算。
- 充分性判定最多触发一轮补检；之后仍缺证据则返回缺失项。回答核验区分引用存在与事实支持，最多一次修订，失败明确返回 `verification_failed`。
- 管理端可用 `trace_id` 查候选、分路排名、耗时、窗口、补检和核验记录。普通响应不返回完整内部追踪；显式范围是可选新增字段，原接口仍可使用。

## 配置与索引

候选配置在 `config/quality-v2.json`，不是已验收的生产配置。初始值为 BM25/向量各 80、重排 60、返回 12、RRF k=10、重排长度 1024、上下文 8000。k=10/30/60 和模型替换必须按开发集对照后决定。

`基于STM32的电机状态监测系统设计_闫丹.pdf` 的物理第 75 页经原图复核为装饰性空白封底；配置仅对这个源哈希的该页使用 text。复核记录和渲染保存在 `.rag/reports/quality-v2/pdf-page-reviews.json`。源文件改变会拒绝复用该覆盖配置，需重新检查。空白正文不会作为检索证据。

解析、分块和 embedding 配置进入索引 fingerprint；模型 revision、维度、输入预处理进入向量缓存 fingerprint。Python 解析缓存单独标记 AST 版本。改变分块、模型或页级覆盖须构建新代次。旧索引没有 fingerprint 时允许兼容读取，禁止在其中混写新块。

PowerShell 示例（项目所用解释器）：

```powershell
$python = 'C:\Users\23245\.conda\envs\langchain\python.exe'
& $python -m pytest tests -q -p no:cacheprovider
& $python -m ragapp --config config/quality-v2.json rebuild
```

重建会生成独立 collection/catalog；全部文件成功后才保存 `.rag/generations/<collection>.json`，不会自动发布。失败保留原索引、任务错误及解析检查点。重复重建使用校验过的页面/向量缓存。全部实测门槛与独立审核通过后，手动使用 `publish <generation.json> <acceptance-report.json>` 原子切换活动指针，重启服务读取它。前一指针保存在 `.rag/active.previous.json`；回滚时停止服务，将它的内容原子写回 `.rag/active.json`，保留两代数据库和缓存。

候选运行配置必须合并代次文件中的 `collection`、`catalog`、`settings`，并保留 `load_active:false`。不要直接用质量配置对旧库执行 `sync`。

## 题集和评测

旧代码、配置、原题与原报告哈希快照位于 `.rag/experiments/quality-v2-baseline/`，原 SQLite 已单独备份。v1题集位于 `.rag/evaluation/quality-v2/`：120 开发、120 冻结盲测，按证据簇和模板隔离，同一源文档可在两组出现。两组由不同子智能体对原文独立核对，记录 `agent_source_reviewed`；不称为人工审核。原图题保留视觉审核记录，不用文本 span 自动判分。新增且已封存的等价证据副本位于 `.rag/evaluation/quality-v2-evidence-v2/`，原问题及其他语义字段不变，v1字节与成绩仍保留。

`scripts/build_quality_evaluation.py refresh` 曾用于v1独立来源审核完成后的冻结；当前已有完整历史报告，不能为调整成绩原地刷新v1或封存后的evidence-v2。后续证据表示修订应新建版本并按 [证据修订操作](evidence-revision.md) 完成审核闭环。题集、源文件、代码、索引或配置改变后必须新建报告，禁止拼接旧运行结果。盲测只用于冻结版本的最终检验；模型对照工具默认拒绝盲测。

以下保留首轮v1复现命令，不能把当前代码续接到已完成的历史报告。当前新实验应显式传入 `--cases .rag/evaluation/quality-v2-evidence-v2/dev.json` 并选择新的报告名；既有首轮响应可用replay重评分，无需重新调用模型。

```powershell
# 原代码/原索引基线；原实现未暴露候选80，报告会明确缺失该项。
& $python scripts/compare_retrieval.py --variant baseline --report .rag/reports/quality-v2/baseline.json
# 保留旧库和40/40/20预算，测试P0及输入预算修复。
& $python scripts/compare_retrieval.py --variant p0 --report .rag/reports/quality-v2/p0.json
# candidate-settings.json须由上一步生成的代次配置组成。
& $python scripts/compare_retrieval.py --variant quality --settings .rag/reports/quality-v2/candidate-settings.json --rrf-k 30 --report .rag/reports/quality-v2/rrf30.json
```

检索报告分别给出真实片段 Recall@10、候选 Recall@80、上下文覆盖、NDCG、旧定位口径、微平均/查询平均/文档平均及置信区间。页码相同或函数相同不能替代证据文本命中；保留符号、大小写及单位差异。

模型候选下载使用固定 revision 和文件 SHA-256。`scripts/benchmark_models.py prepare/embed/pool/rerank` 固定同一分块和候选池，逐模型执行，有内存检查及 worker 锁。采样语料的 smoke 报告只验证实现，不可用于生产模型定型。

当前full-v5的prepare显式绑定evidence-v2开发集和v5 catalog，不使用脚本历史默认的v1题集。后续新建suite同样需传入 `--dataset .rag/evaluation/quality-v2-evidence-v2/dev.json`；已经开始的suite、代码、模型和配置不能原地变更。

模型对照可用 `--resume` 恢复同一题集、模型文件、代码和配置的实验；文档向量分批提交并校验，查询仍重新推理。E5 可显式使用 `--reuse-index-vectors`，仅复用输入及指纹吻合、逐文件校验且抽样重算一致的向量。缓存复用时间不作为新模型编码吞吐；任何配置或代码改变都应使用新报告。

## 验收与费用

最终发布要求冻结盲测 120 题完整、逐题原文与答案独立审核、旧 60 题至少 57 正确；新正确率≥90%、候选 Recall@80≥95%、真实 Recall@10≥90%、支持率≥95%、有效物理引用100%、拒答 precision/recall 各≥90%。热启动完整问答 p95≤60秒，服务与数据库合计峰值 RSS≤12GiB。缺失值、NaN 和不完整报告不能通过。

本轮新增实验费用统一使用 `campaign=quality-v2`，独立台账 `.rag/billing.sqlite3`，累计上限100元，preparation/development/acceptance为20/50/30元；包含本campaign已发生费用，续跑或新建报告不会重置额度。历史账目继续保留；失败调用以已结算额或预留上限计入。人民币数字是按配置单价和保守汇率的估算，最终账单以服务商为准。网络/额度失败保留检查点，不自动切换到另一个付费通道。

2026-10-03 已复核 [DeepSeek 官方价格](https://api-docs.deepseek.com/quick_start/pricing/)：本轮保守估算继续采用 Flash 高峰未命中缓存的输入 $0.3/百万 token、输出 $1.2/百万 token，并按固定10元/美元上界转换；忽略折扣。复核记录在 `.rag/reports/quality-v2/pricing-verification.json`。同日免费通道单次探测仍返回账户试用配额耗尽，记录于 `provider-probe.json`；不能将其写成免费通道验收完成。

发布前还需核对当前代码与源文件未漂移、待发布 catalog 的实际 fingerprint、完整资源记录和成本台账。具体发布报告字段由 `publish_generation` 的契约定义。未达到门槛时保留候选配置和报告，继续使用原索引；不得将未完成的免费通道、人工审核或质量实验写成通过。
