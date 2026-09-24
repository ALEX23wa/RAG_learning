# RAG Learning

这是我学习 **RAG（Retrieval-Augmented Generation，检索增强生成）** 的个人项目，用来整理学习笔记、编写实验代码，并逐步搭建自己的知识库。

我希望通过实际运行和观察，理解一份资料如何经历解析、分块、向量化、检索，最后成为模型回答中的证据。目前从文本型 PDF 开始，后续继续探索多模态检索、不同向量数据库和检索效果评估。

## 当前进度

| 内容 | 状态 |
| --- | --- |
| RAG 原理、系统设计与 LangChain 接口学习 | 已整理学习文档，持续补充 |
| 文本型 PDF 解析 | 已实现逐页提取，保留来源、文件哈希和物理页码 |
| 递归分块 | 已实现，可设置块大小和重叠字符数 |
| 标题层级与父子分块 | 已实现，子块用于检索，完整父章节供生成阶段回取 |
| 分块验证 | 已完成单元测试与实际 PDF 的文本覆盖、偏移和父子关联检查 |
| 向量入库、检索与 LLM 问答 | 待接入当前数据准备流程 |
| Milvus 服务部署 | 已在本地通过 Docker Compose 部署，作为后续实验环境 |
| Visualized-BGE 多模态检索 | 学习与搭建中，`multis.py` 尚未形成完整流程 |

当前代码重点是 **PDF 数据准备**，不调用模型 API，不要求启动数据库；暂不包含 OCR、完整问答链路或 Web 界面。

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
│   ├── data_preparation.py          # PDF 提取、递归分块与父子分块
│   ├── test_data_preparation.py     # 分块与父文档回取测试
│   ├── DATA_PREPARATION.md          # 详细使用说明
│   ├── prepared/                   # 本地生成数据，已被 Git 忽略
│   ├── multis.py                    # 多模态实验草稿
│   └── papers/                     # 实验 PDF 资料
├── .env                            # 本地配置，自行创建，不提交 Git
└── Model/                          # 本地服务配置和实验资源，已被 Git 忽略
    └── docker-compose.yml          # Milvus、etcd、MinIO
```

运行后会在 `prepared/` 下生成逐页文本、文本块和处理记录。`Model/` 是本地目录，不随仓库克隆获取；数据准备脚本不依赖其中的 Docker 服务。

## 学习文档

- [PDF 数据准备说明](rag-practice/DATA_PREPARATION.md)：当前脚本的参数、输出格式与父文档回取示例。
- [RAG 知识学习手册](docs/RAG知识学习手册.md)：从基础概念到代码实践，包含实验步骤与排错说明。
- [RAG 系统构建方案](docs/RAG系统构建方案.md)：记录个人知识库的架构、数据流、技术选型和后续设计。
- [LangChain 1.x 接口使用手册](docs/LangChain1.x接口使用手册.md)：查阅常用接口、输入输出和调用示例。

建议先运行下面的数据准备流程，再结合知识学习手册和系统构建方案扩展。学习手册中仍保留早期问答示例，相关历史脚本已移除；当前可执行入口与支持能力以本 README 和数据准备说明为准。

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

### 2. 提取 PDF 并选择分块方式

将文本型 PDF 放入 `rag-practice/papers/`，脚本会递归扫描其子目录。默认输入、输出路径按脚本位置定位，不依赖终端工作目录。当前脚本无需密钥，也不读取 `.env`。

```powershell
python .\rag-practice\data_preparation.py --help

# 递归分块：每块最多 800 字符，重叠 50 字符
python .\rag-practice\data_preparation.py --strategy recursive --chunk-size 800 --overlap 50

# 结构分块：完整二级章节作为父文档，检索子块最多 300 字符
python .\rag-practice\data_preparation.py --strategy structure --parent-level 2 --child-size 300 --overlap 50
```

结构模式为默认模式。`--parent-level 2` 以 `3.2` 等标题划分父章节，包含其下 `3.2.1` 等小节；设置为 `3` 可得到更细的父章节。标题识别优先使用 PDF 书签及其正文位置，没有可用书签标题时尝试正文编号。未识别出标题时，以整份提取文本作为父文档，并记录提示。

父章节可以跨页，保持完整内容。扫描页需要另行 OCR；多栏、表格和公式的提取顺序仍需对照原文检查。

### 3. 查看输出并回取父文档

每次运行都会生成独立批次目录，具体路径打印在终端：

```text
rag-practice/prepared/<strategy>/<batch-id>/
├── pages.jsonl       # 逐页文本、物理页码与字符偏移
├── children.jsonl    # 用于后续嵌入和检索的文本块
├── parents.jsonl     # 结构模式的完整父章节；递归模式为空
└── manifest.json     # 处理参数、数量及提取警告
```

后续接入检索时，向量化 `children.jsonl` 中的 `page_content` 并保留元数据。结构模式根据命中子块的 `metadata.parent_id`，从同一批次的 `parents.jsonl` 回取父章节；`build_parent_context()` 会去重并组织完整上下文，供调用方传给 LLM。

父文档可能超过模型上下文限制。该函数支持 `max_chars` 字符预算，超限时报错，不会静默截断；此时应减少回取父文档数、选择更长上下文的模型，或提高 `--parent-level` 后重新分块。字符预算不等于 token 预算。

具体调用方式见 [父文档回取示例](rag-practice/DATA_PREPARATION.md)。当前脚本仅准备数据，不执行向量入库、检索或 LLM 调用。

### 4. 运行测试

```powershell
python -m unittest discover -s rag-practice -p test_data_preparation.py
```

测试覆盖同页多个标题、跨页父章节、父子关联、上下文去重与预算、无标题回退和非法参数。

当前 IPCC 样例的处理记录（默认参数）：

| 模式 | PDF 页数 | 父文档数 | 文本块 / 子块数 |
| --- | --- | --- | --- |
| 递归分块 | 172 | 0 | 1,479 |
| 标题层级分块 | 172 | 24 | 4,204 |

两种模式均完成字符偏移和非空白文本覆盖检查，结构模式额外检查父子关联。修改 PDF 或分块参数后，数量可能变化；这些检查不等于已验证检索效果。

## 技术与实验方向

当前数据准备使用 **Python、pypdf 与 LangChain 的递归文本分割器**，以 JSONL 保存结果。下一步将子块接入向量库，完成“检索子块 → 回取完整父文档 → 生成回答”的流程。

本地已准备 Milvus、etcd 和 MinIO 服务。本机已有 Compose 文件时，可从项目根目录运行：

```powershell
docker compose -f .\Model\docker-compose.yml up -d
docker compose -f .\Model\docker-compose.yml ps
```

本地 Milvus 端口为 `19530`，MinIO 控制台端口为 `9001`。服务启动后仍需创建集合并导入向量；相关代码尚未与数据准备串联。

`multis.py` 包含 Visualized-BGE 编码器草稿与 Milvus 连接配置，还不是完整的多模态检索程序。模型路径需要按本机调整，Visualized-BGE 子包与权重需单独准备。

## 后续学习计划

- [x] 完成 PDF 提取、递归分块和标题层级父子分块。
- [ ] 将子块嵌入并写入向量库，接通父文档回取与 LLM 问答。
- [ ] 比较分块大小、重叠长度和检索数量对结果的影响。
- [ ] 建立小型问题集，分别评估检索命中、答案依据和拒答表现。
- [ ] 尝试关键词与向量混合检索、重排和查询改写。
- [ ] 完成 Visualized-BGE 与 Milvus 的多模态检索实验。
- [ ] 探索扫描 PDF 的 OCR、增量更新和简单交互界面。

## 资料与配置管理

密钥放在本地 `.env` 中，不写入代码或提交到 Git。上传资料前确认分享权限；生成的索引可能包含原文内容，也应作为本地数据管理。提交前检查 `git status`，避免把密钥、私人资料、模型权重或运行产生的索引一并上传。

这个仓库会随着学习不断调整，保留实验过程中的理解、问题和改进记录。
