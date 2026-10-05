# 普通候选索引构建与恢复

`scripts/rebuild_candidate.py` 使用正常 `Service` / `Indexer.sync()`，重新读取原资料、复用有来源校验的解析缓存、按候选 tokenizer 分块并生成向量。它不复制 benchmark 冻结块，不读取开发题、盲题或排名，不修改 `active.json`，也不自动发布。完成代次仍须单独评测与验收。

## 本轮 BGE 命令

先结束其他模型实验、导入和同步任务，并确认既有本机 Qdrant 正常运行。脚本不会为预检启动数据库；集合清单读取失败时退出。以下仅是调用说明，文档本身不执行构建。

```powershell
& 'C:\Users\23245\.conda\envs\langchain\python.exe' scripts/rebuild_candidate.py `
  --settings .rag/reports/quality-v2-bge-live-v1/candidate-settings.json `
  --source-settings .rag/reports/quality-v2/candidate-input-v5-settings.json `
  --report .rag/reports/quality-v2-bge-live-v1/rebuild.json `
  --expected-ready 38 --rss-gib 12 --allow-vision
```

失败后先检查报告原因。修复外部暂时性故障后，使用完全相同命令，末尾追加 `--resume`。脚本不自动重试、不每次生成新的集合名、不删除失败现场。正常同步按文档提交；此前成功且版本一致的文档由 `Indexer.sync()` 跳过。中断文档可能复用有效解析缓存，仍需完成自身正常入库。

## 输入与归属

- 两个 settings 文件都是完整 `Settings` JSON，必须显式含 `collection`、`catalog`。相对目录沿用 `ragapp.experiments.load_settings()` 的规则，以各设置文件所在目录为基准。强制 `load_active=False`。
- 源库必须对应 `state/generations/<源 collection>.json`，配置与其设置一致，SQLite 元数据的指纹与代次文件一致。源库只通过 SQLite `mode=ro` 读取。
- 候选与源共用资料根目录、state、Qdrant 及既有 campaign/billing.sqlite3。不可提高原 campaign 及阶段额度，总额度不超过 100 元。
- 拒绝源集合、源 catalog、活动集合/catalog、默认旧库和 billing.sqlite3。首次运行也拒绝任意已存在的目标集合、catalog、归属文件或 generation 文件。
- 先独占写入 `--report` 与 `state/candidate-builds/<collection>.owner.json`，再创建 Service、候选 SQLite 和 Qdrant 集合。候选 metadata 保存相同归属。源排除状态、calls、call_campaigns、jobs、traces 和其他历史 metadata 会迁移；源 chunks 与旧 pipeline 指纹/schema/profile 不会迁移。
- 恢复要求同一报告、build_id、job_id、目标集合/catalog、配置原字节、代码、包版本、Python、模型原文件、原资料哈希、源台账逻辑内容、源代次、活动指针和视觉策略。缺失或被替换的资源不会被静默重建。

## 进程、费用和进度

外层同时持有 `server.lock`、`candidate-build.lock`、`reports/quality-v2/experiment-worker.lock`、`reports/quality-v2-models/model-worker.lock` 和报告专属锁，全部 `timeout=0`。预检同步锁后释放，再由正常 `Indexer.sync()` 持有同步锁，避免嵌套死锁。另检查已知模型工作进程及报告中的 PID/创建时间；不停止任何无关进程。

报告字段：

| 字段 | 含义 |
| --- | --- |
| `status`, `phase`, `updated_at_utc` | 当前状态、阶段及最后检查点时间 |
| `worker_pid`, `worker.create_time` | 实际 worker PID 与进程创建时间，避免 PID 重用误判 |
| `collection`, `catalog`, `owner.report` | 固定目标及归属报告 |
| `job_id`, `job.status`, `job.progress` | 正常 Store 任务；progress 保留其原 JSON 字符串格式 |
| `sync_result` | 本次完成、跳过、失败及具体文件错误 |
| `attempts` | 各次尝试的起止时间、状态、异常与资源报告 |
| `historical_peak_rss_bytes` | 所有已持久化尝试的最高 RSS，恢复不会清零 |
| `verification` | 实际 ready 文档数、当前版本块数及向量数 |
| `resources_verified` | 是否有真实有效且历史完整的资源测量；模拟依赖恒为 false |

默认 `cache-only` 仍让索引器读取已有图片描述缓存，缺失时该文档明确失败，不能以 `models=None` 悄悄漏掉描述。`--allow-vision` 仅放行正常索引器已有的 `production` / `preparation` 调用，沿用共享 campaign 台账、预算预留与计费；缓存存在时仍复用。它不会切换付费 profile，也不授权额外任务。该参数属于冻结身份，不能在一次 cache-only 构建失败后直接改变同一报告的策略；本轮已有授权，因此首跑即显式传入。

## 完成与资源限制

只有原资料集合中所有非排除文档均 ready、原文哈希一致、每文档存在当前版本块、无失效块、向量数与块数一致、来源与代码/模型身份未变化、本次资源测量有效且没有已知历史超限，才写入 `state/generations/<collection>.json`。`--expected-ready 38` 是额外约束，不会掩盖真实资料数量变化。

RSS 使用既有 `ResourceMonitor`，统计 worker、其子进程和 Qdrant，每个正常任务进度边界检查已锁存的峰值，最大预算 12 GiB。它是采样和边界中止，不是操作系统硬内存限制；长解析或模型操作期间的短暂超限会在下个边界报告失败。零值、不可用测量不能成为通过结果，任何已记录超限也不能靠 resume 清零。

正常异常与 Ctrl+C 会持久化失败和资源记录。强制终止/掉电无法证明最后一次检查点之后的峰值；恢复后保留 `interrupted_without_final_measurement`，即使所有文档后来完成，`resource_history_complete=false`、`resources_verified=false`，代次文件也记录该限制，不能据此声称整个构建历史通过资源验收。已知历史峰值仍保留。此工具从不设置 `publication_eligible=true`。

## 验证范围

```powershell
& 'C:\Users\23245\.conda\envs\langchain\python.exe' -m pytest tests/test_rebuild_candidate.py -q
```

测试仅使用临时语料/SQLite、正常 Indexer、模拟向量/模型/资源采样，覆盖断点继续、不可变身份、未知目标保护、锁、历史迁移、图片描述、资源异常和完成条件。模拟结果明确标为 `complete_synthetic`、`synthetic_dependencies=true`，代次带 `synthetic=true`；不能充当真实模型或内存验收。
