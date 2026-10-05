# RAG Learning

面向 PDF、Word、Excel、CSV、图片、代码和 Markdown 的本地知识检索系统，提供网页、API 和命令行入口。

本轮质量优化已完成并收尾。最佳完整开发轮次严格审核 **117/120（97.5%）**，旧 60 题回归 **59/60**。新 120 题盲测严格审核 **89/120（74.17%）**，召回、拒答和延迟未达到原发布门槛，因此保留原生产索引。完整结果、计分口径及限制见 [结项报告](docs/quality-v2-status.md)；开发成绩不能代替盲测成绩。

## 运行

```powershell
conda activate langchain
python -m pip install -r requirements-rag.txt
python scripts/bootstrap_rag.py
python -m ragapp serve
```

浏览器打开 <http://127.0.0.1:8000>，使用本机 `.rag/tokens.json` 的 `admin` 令牌。原始资料放在 `papers/`，模型、索引、运行记录和费用台账放在 `.rag/`。这些本地数据和 `.env` 不随本次代码提交上传。

首次安装和 MinerU 配置见 [环境安装](docs/本机环境安装与MinerU.md)；日常配置、权限、备份与恢复见 [运行手册](docs/RAG运行与验收.md)。不要把管理令牌或服务商 API 密钥写入代码。

## 实现

- 明确文档范围才做硬过滤；主题词只用于软召回，当前问题始终保留。
- BM25、向量与定位候选统一融合，模型输入有完整 tokenizer 预算检查。
- 按资料结构分块，保留原文、父节、字符范围、表格坐标和代码行号。
- 重排后合并重叠证据、扩展父节；最多一次补检与一次有依据的回答修订。
- CSV / Excel 使用受限计算器，原始数字、单位和同行身份可追溯。
- 代码、语料、模型和配置冻结后评测；缓存隔离、断点校验、计费与发布门槛可复核。

最佳实验使用 BGE-M3 embedding 和 BGE reranker，参数见 [quality-v2-best.json](config/quality-v2-best.json)。它是**复现实验参数**，不是已发布配置；移除了本机绝对路径及索引名称。必须先构建独立代次，再用生成的代次信息配置运行，不能直接对旧库执行 `sync`。最初 E5 对照参数保留在 `config/quality-v2.json`。

## 项目目录

| 路径 | 内容 |
|---|---|
| `ragapp/` | 服务、检索、解析、计算、核验、评测和索引维护 |
| `data_process/` | PDF 文本提取与 MinerU 接入 |
| `scripts/` | 安装、构建、冻结、评测及对照工具 |
| `tests/` | 工程回归测试 |
| `config/` | 基线和最佳实验参数 |
| `docs/` | 结项报告与操作文档 |
| `docs/archive/` | 历史实验与阶段记录，不代表最终状态 |

验证本轮冻结运行代码的全量回归为 **972 项通过**。测试验证工程行为；答案质量以独立来源审核和实测报告为准。

```powershell
python -m pytest tests -q -p no:cacheprovider
```

详细入口：[实现与索引切换](docs/RAG质量优化实施.md)、[评测与断点恢复](docs/quality-evaluation.md)、[历史实验索引](docs/archive/README.md)。本轮费用累计 **47.718267 元 / 100 元**，包含失败调用和保守预留；按用户要求不再追加实验。免费通道历史仅完成 3/60，未完成验收。
