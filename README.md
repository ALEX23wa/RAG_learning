# RAG Learning

这是我学习 **RAG（Retrieval-Augmented Generation，检索增强生成）** 的个人项目，用来整理学习笔记、编写实验代码，并逐步搭建自己的知识库。

我希望通过实际运行和观察，理解一份资料如何经历解析、分块、向量化、检索，最后成为模型回答中的证据。目前从文本型 PDF 开始，后续继续探索多模态检索、不同向量数据库和检索效果评估。

## 当前进度

| 内容 | 状态 |
| --- | --- |
| RAG 原理、系统设计与 LangChain 接口学习 | 已整理学习文档，持续补充 |
| 文本型 PDF 解析与分块 | 已实现逐页解析，保留来源和物理页码 |
| 本地向量检索 | 已实现 Qdrant 本地存储与 SQLite 内容记录 |
| 离线流程演示 | 已实现哈希特征向量检索，无需模型 API |
| API 问答 | 已实现模型接入、证据引用与引用 ID 校验，实际运行需配置服务 |
| Milvus 服务部署 | 已在本地通过 Docker Compose 部署，作为后续实验环境 |
| Visualized-BGE 多模态检索 | 学习与搭建中，`multis.py` 尚未形成完整流程 |

这是学习中的实验项目。当前 PDF 示例尚不支持 OCR、混合检索、重排、Web 界面或完整的增量更新。

## 项目结构

```text
RAG_learning/
├── README.md
├── environment.yml                 # Conda 环境与依赖记录
├── docs/
│   ├── RAG知识学习手册.md
│   ├── RAG系统构建方案.md
│   └── LangChain1.x接口使用手册.md
├── rag-practice/
│   ├── rag_demo.py                  # 文本 PDF 入库与检索示例
│   ├── multis.py                    # 多模态实验草稿
│   └── papers/                     # 实验 PDF 资料
├── .env                            # 本地配置，自行创建，不提交 Git
└── Model/                          # 本地服务配置和实验资源，已被 Git 忽略
    └── docker-compose.yml          # Milvus、etcd、MinIO
```

运行后会在配置的状态目录下生成向量索引、SQLite 数据库和索引版本信息。`Model/` 是本地目录，不随仓库克隆获取；运行 `rag_demo.py` 无需启动其中的 Docker 服务。

## 学习文档

- [RAG 知识学习手册](docs/RAG知识学习手册.md)：从基础概念到代码实践，包含实验步骤与排错说明。
- [RAG 系统构建方案](docs/RAG系统构建方案.md)：记录个人知识库的架构、数据流、技术选型和后续设计。
- [LangChain 1.x 接口使用手册](docs/LangChain1.x接口使用手册.md)：查阅常用接口、输入输出和调用示例。

建议先阅读知识学习手册并运行 PDF 示例，再结合系统构建方案逐步扩展；编写代码时按需查阅接口手册。文档中的规划不代表当前代码已全部实现。

## 快速开始

以下命令使用 Windows PowerShell，起始位置为项目根目录。

### 1. 使用 Conda 环境

本项目使用名为 `langchain` 的 Conda 环境：

```powershell
conda activate langchain
python -c "import sys; print(sys.executable)"
```

本机解释器路径为 `D:\Anaconda\envs\langchain\python.exe`。其他机器以自己的 Conda 安装位置为准。

如果尚未创建环境，可以先检查 `environment.yml` 中的平台依赖和末尾的 `prefix`，调整为本机路径后执行：

```powershell
conda env create -f environment.yml
conda activate langchain
```

`environment.yml` 是学习环境的完整记录，包含 PDF 示例以外的依赖，并非跨平台的最小安装清单。

### 2. 配置离线模式

在项目根目录创建或编辑 `.env`，合并以下配置，不要覆盖已有密钥或其他配置：

```dotenv
RAG_MODE=demo
RAG_STATE_DIR=./.rag_state
```

脚本按自身位置查找根目录 `.env`，已有系统环境变量优先。`RAG_STATE_DIR` 的相对路径以运行命令时的工作目录为准，因此下面统一在 `rag-practice` 目录执行。

离线模式使用哈希特征模拟向量，不调用模型服务，也不生成答案，适合检查资料解析、索引和检索流程，不能用来衡量真实语义检索效果。

### 3. 导入 PDF 并检索

将可用于实验的文本型 PDF 放入 `rag-practice/papers/`，然后执行：

```powershell
Set-Location .\rag-practice
python .\rag_demo.py --help
python .\rag_demo.py ingest .\papers
python .\rag_demo.py ask "这份资料讨论了哪些主要问题？" --k 4 --search-only
```

请把问题替换成资料中能够找到证据的问题。入库成功时输出 `READY`，检索结果会显示候选原文、文件路径和物理页码。

每次 `ingest` 都会重新构建指定目录的完整快照，成功后切换当前索引。修改或移除 PDF 后需要重新入库；旧索引版本仍会保留在磁盘上。

### 4. 切换到真实模型问答

将根目录 `.env` 中对应的配置改为：

```dotenv
RAG_MODE=api
RAG_STATE_DIR=./.rag_state_api
OPENAI_API_KEY=替换为自己的密钥
OPENAI_BASE_URL=https://api.openai.com/v1
RAG_EMBED_MODEL=text-embedding-3-small
RAG_CHAT_MODEL=gpt-4.1-mini
RAG_ALLOW_REMOTE=1
```

模型名称和地址需与你使用的服务匹配；兼容服务还需支持脚本使用的结构化输出。仅在确认资料允许发送到模型服务后启用 `RAG_ALLOW_REMOTE=1`：入库会发送文本块，提问会发送问题和检索证据，并可能产生调用费用。

在 `rag-practice` 目录重新入库后提问：

```powershell
python .\rag_demo.py ingest .\papers
python .\rag_demo.py ask "这份资料讨论了哪些主要问题？" --k 4
```

回答会尝试引用 `[S1]` 等证据编号，也可能返回证据不足或引用不合法的提示。引用编号检查通过并不代表答案一定正确，仍需回到原文核对。

## 技术与实验方向

当前 PDF 示例使用 **Python、LangChain、pypdf、Qdrant 和 SQLite**。Qdrant 以本地模式运行，SQLite 保存来源与文本块。API 模式通过 `langchain-openai` 接入嵌入模型与生成模型。

Milvus 是另一条学习实验路线，目前尚未接入 `rag_demo.py`。本机已有 Compose 文件时，可从项目根目录运行：

```powershell
docker compose -f .\Model\docker-compose.yml up -d
docker compose -f .\Model\docker-compose.yml ps
```

本地 Milvus 端口为 `19530`，MinIO 控制台端口为 `9001`。服务启动后仍需由后续代码创建集合和导入向量。

`multis.py` 用于探索 Visualized-BGE、图片处理与 Milvus，目前仅有导入和初始配置，模型路径也需要调整。它还不能直接完成图片入库或多模态检索。Visualized-BGE 的子包和权重需另行准备，不能仅凭安装 `FlagEmbedding` 主包就直接运行。

## 后续学习计划

- [ ] 比较分块大小、重叠长度和检索数量对结果的影响。
- [ ] 建立小型问题集，分别评估检索命中、答案依据和拒答表现。
- [ ] 尝试关键词与向量混合检索、重排和查询改写。
- [ ] 完成 Visualized-BGE 与 Milvus 的多模态检索实验。
- [ ] 探索扫描 PDF 的 OCR、增量更新和简单交互界面。

## 资料与配置管理

密钥放在本地 `.env` 中，不写入代码或提交到 Git。上传资料前确认分享权限；生成的索引可能包含原文内容，也应作为本地数据管理。提交前检查 `git status`，避免把密钥、私人资料、模型权重或运行产生的索引一并上传。

这个仓库会随着学习不断调整，保留实验过程中的理解、问题和改进记录。
