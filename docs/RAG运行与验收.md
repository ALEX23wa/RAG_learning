# 本机多格式 RAG

应用为 Python，使用 Conda `langchain`，Qdrant 官方 Windows 1.19.1 二进制和 SQLite FTS5。原文件在 `papers`，运行数据全部在 `.rag`，不要提交原文件、密钥、令牌、私有测试答案或缓存。

本页保留部署方法及旧版交付记录；本轮优化已经结束，最终成绩、预算及未发布原因见 [结项报告](quality-v2-status.md)。

## 安装与运行

```powershell
conda activate langchain
python -m pip install -r requirements-rag.txt
python scripts/bootstrap_rag.py
python -m ragapp doctor
python -m ragapp serve
```

网页：<http://127.0.0.1:8000>。服务首次运行创建 `.rag/tokens.json`，将 `admin` 的值填入网页管理令牌输入框。仅监听本机。启动扫描、文件监听和 60 秒核对清单会同步数据。首次解析大型扫描资料可能需要较长时间，可在资料状态中查看当前文件、错误及已完成数量。

现有解释器路径为 `C:\Users\23245\.conda\envs\langchain\python.exe`，已迁回 Conda 用户默认环境目录。MinerU 权重继续使用 `D:\Program Files\MinerU\models`；RAG 向量和重排模型位于项目 `.rag/models`。AMD RX 6650 XT 使用 MinerU 原生 llama.cpp Vulkan，不能用 `torch.cuda.is_available()` 判断该后端。小型 ONNX 组件与 embedding/reranker 使用 CPU。

```powershell
python -m ragapp sync
python -m ragapp search "问题"
python -m ragapp ask "问题"
python -m ragapp ask "问题" --profile eval_aihub_free
python -m ragapp backup D:\backups\papers-rag.zip
# 停止网页服务后执行恢复或建立候选索引；rebuild 不发布：
python -m ragapp restore D:\backups\papers-rag.zip
python -m ragapp rebuild
```

备份包含 SQLite 台账、Qdrant 快照和索引引用的图片资产，并校验资产哈希；不包含原始 `papers`、模型、`.env`，它们需要单独保留。恢复会迁移图片路径，并保留本机已有的付费调用记录和免费请求限速状态，不能通过恢复重置预算。重建只写独立候选集合和台账，全部文档成功也不会自动切换 `.rag/active.json`。满足完整实测与独立审核门槛后，再手动执行 `publish`；原集合和前一活动指针保留用于回滚，详见 [实施说明](RAG质量优化实施.md)。

## 模型与费用

`.env` 使用 `DEEPSEEK_API_KEY` 与 `AIHUB_API_KEY`。`production` 的回答和视觉复核使用 DeepSeek `deepseek-flash`。`eval_aihub_free` 只使用 `https://aihubmix.com/v1` 的免费文本别名 `coding-glm-5.1-free` 和视觉模型 `agents-a1-free`，没有付费回退。实际返回模型可能与别名不同，逐次记录 requested_model 与 actual_model。

免费调用全局至少间隔 13 秒；429 保存 Retry-After 和评测断点，不计为通过。账户免费额度与模型可列出是两回事。官方免费标价/能力核对地址：

- <https://aihubmix.com/model/coding-glm-5.1-free>
- <https://aihubmix.com/model/agents-a1-free>
- <https://aihubmix.com/models/free>

本轮quality-v2新增实验费用的累计上限为100元，准备20、开发50、验收30元，统一记录到 `campaign=quality-v2` 的 `.rag/billing.sqlite3`。该额度包含本campaign已经发生的调用，续跑不会重置；旧账继续保留。旧版无campaign路径仍使用30元总额及5/10/15元阶段限制，这不是本轮实验预算。按已核对的Flash输入0.3 USD/百万、输出1.2 USD/百万和保守的10 CNY/USD换算，不利用缓存/低峰折扣。调用前预留，成功后按usage结算，无usage的失败保留预留；这是估算而非平台账单。价格复核时间及记录见 [实施说明](RAG质量优化实施.md)。

## API 与权限

除 `/api/health` 外，API 需要 `Authorization: Bearer <token>`。

| 令牌 | 模型与权限 |
|---|---|
| admin | production；检索、问答、同步、状态、受控原文、排除文档、用量 |
| blackbox | production；仅检索和问答 |
| eval_aihub_free | eval_aihub_free；仅检索和问答 |

POST `/api/search` 和 `/api/chat` 请求为 `{"question":"问题","history":[]}`，history 可包含此前用户问题，但历史回答不会当作事实。服务端固定模型配置，客户端不能传任意端点。POST `/api/sync` 返回 job_id；GET `/api/jobs/{id}` 查看进度。GET `/api/documents` 列表，GET `/api/documents/{id}/source?version=...` 校验版本并下载，GET `/api/documents/{id}/preview` 预览文本定位。DELETE `/api/documents/{id}` 持久排除但保留磁盘原件。

引用绑定文档版本与页、段落、工作表范围或代码行。检索复核活跃版本；写入失败保留旧版本并标记 `update_failed`。原文预览在内容变化后拒绝返回错误版本。解析缓存按内容哈希复用，向量按文本哈希复用。扫描排除派生 `extracted`、隐藏文件、临时文件和符号链接。

计算器支持 CSV/XLSX 的 sum/mean/min/max/count 和显式列过滤，不执行模型生成代码。问答对明确的 CSV 列名与聚合操作可自动调用，也支持明确的 ISO 日期范围或单一年份；含糊条件不会擅自计算。管理令牌还可调用 `/api/calculate`，提供 document_id、operation、column、sheet 和 filters。缺失 Excel 公式缓存会报错，不能声称已重算公式。汇总引用显示参与计算的原表行范围。

## 测试与验收

```powershell
python -m pytest tests -q
python scripts/stress_rag.py
python -m ragapp evaluate .rag/evaluation/development.json .rag/reports/development.json --profile production --stage development
python -m ragapp evaluate .rag/evaluation/aihub_acceptance.json .rag/reports/aihub.json
```

上面的evaluate命令指向历史评测集：40道开发题与两组各60道验收题，旧目标为51/60、旧口径Evidence Recall@10≥85%、事实支持率≥90%、引用定位有效率100%；这些目标仅解释旧报告，不能作为本轮发布门槛。当前quality-v2使用120开发+120盲测及封存的evidence-v2，旧60题仅作为同运行版本下至少57/60的回归要求。当前正式门槛及完整问答操作见 [评测说明](quality-evaluation.md#独立审核和发布)。标准答案不能进入索引；自动正则/数字评分不能替代逐主张、数字、原文和引用核验，未审报告不会自动通过。

Codex 黑盒子智能体仅收到第二组问题和受限令牌，只能调用 HTTP 搜索和问答，不能读取原文件、索引、缓存、密钥或标准答案。这是行为和接口权限约束，不是操作系统沙箱。违反约束则测试作废。

扩容脚本在 `.rag/scale` 和独立 6335/6336 端口验证 100,000 个 768 维合成向量的写入、删除、重启及内存，报告 `.rag/reports/scale-100k.json`。它只验证存储能力，不代表十万文本块的语义准确率或端到端延迟。

验收结果以 `.rag/reports` 实际报告为准；资源下载成功、单元测试成功或模型调用成功均不代表问答准确率已经达标。

## 旧版交付记录（2026-10-03保留，不代表本轮状态）

旧交付报告记录真实语料38份、PDF共899页、原索引12,755块。Codex子智能体通过受限接口完成60题，主会话对照原文件复核：57/60、旧口径Evidence Recall@10 92.98%、事实支持率97.81%、引用物理定位有效率100%。三个失败回答与全部原始输出继续保留。以上是旧报告记录，不是v5、evidence-v2或本轮最终盲测成绩；原报告未在此标注执行日期，不能把本页保留日期视为运行日期。

AIHUB 第一轮仅完成 3/60 个有效回答，第 4 题起账户返回“未充值账户只能试用 10 次”。该提示以 HTTP 200 返回，现已增加错误识别和回归测试；原始错误记录、费用台账更正记录及断点均保留。**整体双通道验收尚未完成**。不能把部分题目的分数当作整轮成绩，也不会自动充值或使用付费替代模型。账户具备可用免费额度后，恢复同一轮：

```powershell
python -m ragapp evaluate .rag/evaluation/aihub_acceptance.json .rag/reports/aihub-acceptance-01.json
```

完整交付摘要在 `.rag/reports/delivery-summary.md`，逐题报告、独立审核、真实调用与费用在同目录。备份为 `.rag/backups/delivery-01.zip`，不含原始资料和模型。

旧版已记录的边界：`.doc`需Word/LibreOffice转换，本机当时未安装可用转换器；`.xls`当时没有真实资料验收。当前候选PDF自动分流已增加文字质量与多栏布局信号，并支持按源哈希绑定的物理页覆盖；仍需核查复杂数学、表格、图注及引用映射。原图可按需渲染复核。有限资料与题集的成绩不代表任意领域或新增资料都能维持同样准确率。
