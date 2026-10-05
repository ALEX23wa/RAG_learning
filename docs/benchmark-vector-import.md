# BGE-M3 候选代次的精确向量复用

本工具只减少新候选索引中重复文本的编码工作，不发布索引，不复制 catalog，不改变旧 E5 指针，也不读取问题、金标或模型排序结果。普通 `rebuild` 仍使用 BGE tokenizer、官方输入模板及配置预算重新分块；新旧分块数可能不同，须重新做端到端评测。

## 调用

在没有服务、构建或其他基准模型进程运行时执行。工具先获取与 benchmark 相同的 `.rag/reports/quality-v2-models/model-worker.lock`，再取 `server.lock` / `sync.lock`，所有锁均 `timeout=0`；冲突立即退出，在任何 tokenizer/模型加载前生效：

```powershell
& 'C:\Users\23245\.conda\envs\langchain\python.exe' scripts/import_benchmark_vectors.py `
  --state .rag `
  --suite <原完整语料-suite目录> `
  --source-cache .rag/reports/quality-v2-models-v2/bge-m3-full-v5.vectors `
  --output .rag/reports/quality-v2-bge-live-v1/vector-import.json `
  --sample-size 32 --batch-size 4 --input-tokens 480 --rss-gib 12
```

源目录必须是 `document-window-cache-v1`，带旧 v3 合同的 BGE-M3 缓存；工具直接读取该合同，绝不构造新 v4 合同冒充原身份。模型限定 `BAAI/bge-m3`，revision `5617a9f61b028005a4858fdac845db406aefb181`，1024 维。实际本地模型及 tokenizer 的全部清单字节必须与原合同和 pinned snapshot manifest 匹配，额外执行文件也会被拒绝。

只读取 suite 的 `manifest.json` 与 `corpus.json`。每个已提交 batch 的文件哈希、元数据哈希、逐向量哈希、唯一键、float32 存储类型、维度、有限性和单位范数全部核验；未提交批次或缺失语料窗口直接拒绝。每块按原 512-token、32-overlap、8-token safety 合同重建窗口，只有单窗口且完整生产输入 UTF-8 字节完全一致、满足目标完整输入 480 token 的条目可导入。被窗口截掉首尾空白、分成多窗口或超目标预算的条目跳过，交给普通构建重新编码。

## 精度与证据范围

历史缓存的计算参数 dtype 未记录，必须保持 `historical_actual_dtype: unknown`。存储数组为 float32 并不能证明历史计算使用 FP32。

同一次显式 CPU FP32 加载会检查所有参数 dtype，以 `(完整输入 token 数, 文本 SHA256)` 排序均匀抽取至少 32 个不同输入重新计算，批次最多 4。比较采用 `rtol=1e-4`、`atol=2e-5`，原始与新向量均要求单位范数绝对误差不超过 `2e-4`。少于所需样本、任何不一致或资源守卫错误均使调用失败；验证完成前不写入新向量。资源守卫监控 worker、子进程与 Qdrant，12 GiB 是上限，检查点间的同步推理仍可能短暂越线后才终止。

成功证明仅是 `sampled_fp32_consistency`，不是全部历史计算精度认证，也不是质量或发布通过。导入耗时不能作为模型编码吞吐。

## 产物与恢复

- 报告 `schema: benchmark-vector-import-v1`，包含 `status`、`run_fingerprint`、`documents_total`、`eligible_unique`、`skipped`、`committed`、`target_folder`、`proof_fingerprint` 和目标 embedding 配置。`publication_eligible` 始终为 false。
- 目标目录是 `.rag/cache/vectors-<FP32 embedding fingerprint>/`。每条 NPY 都有 `verified-document-vector-v1` JSON sidecar，绑定完整输入字节、模型文件指纹、向量校验和及来源。
- `import-<proof fingerprint>.json` 保存旧合同、旧代码/包版本、全部批次校验和、32+ 次样本记录及其限制。没有完整证据的导入缓存不能被 Service 读取。
- 相同命令可以重试。源码、模型、源批次、目标配置或样本参数变化会拒绝续写同一报告。已经完成的精确同向量不会覆盖；冲突在导入前预检。崩溃遗留只有 NPY 的条目只能在重新验证后、与源向量逐字节相等时补齐 sidecar。
- 报告失败时退出非零，保留已经验证和提交的条目及失败报告；下次重试重新核验、重新抽样。报告 `complete` 才表示本次全部合格条目已经处理。写入中若系统阻止原子替换，保留唯一临时检查点并明确报错，最多短暂重试五次。

Service 新候选须设置 `embedding_precision: float32`，并使用相同模型 revision、维度、480/512 预算，正常 `rebuild` 会精确命中这些文本缓存，新增边界或装饰变化则正常编码。缓存读取检查原始向量，查询始终现场计算。模型加载检查参数精度，完整输入超预算直接报错。

默认值 `legacy` 保留历史 embedding fingerprint 的原始 payload，不给旧指纹增加任何字段。旧 active pointer 或 generation manifest 缺少精度字段时明确解释为 legacy，不能继承调用方候选配置的 float32。新 FP32 分支形成独立缓存与索引指纹；旧 E5 collection/catalog/cache 保持原样，可继续查询及回滚。验收发布机制继续独立执行。
