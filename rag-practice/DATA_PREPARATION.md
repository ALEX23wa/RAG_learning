# PDF 文本提取与分块

在项目根目录激活 `conda activate langchain`。默认输入是脚本旁的 `papers/`，不受终端工作目录影响，递归查找所有 `.pdf` / `.PDF` 文件。

## 两种模式

```powershell
# 普通递归分块：每块最多 800 字符，重叠 50 字符
python .\rag-practice\data_preparation.py --strategy recursive --chunk-size 800 --overlap 50

# 结构分块：完整二级章节作为父文档，子块最多 300 字符
python .\rag-practice\data_preparation.py --strategy structure --parent-level 2 --child-size 300 --overlap 50
```

结构模式默认启用。`--parent-level 2` 将 `3.2` 这类标题作为父章节边界，`3.2.1` 等小节保留在该父章节内部；设置为 `3` 会得到更细的父章节。父文档指完整章节，不是固定长度的较大文本块；它可以跨页，不会因 `--child-size` 被截断。标题前的内容也会保留。

程序优先读取 PDF 书签，匹配目标页中的标题原文以确定字符边界，支持同页多个标题。未匹配的书签会记录警告，合并保留其文本而不猜测边界。完全没有可用书签标题时，尝试编号标题（如 `1.2`、`1.2.3`、`第一章`）；仍未识别时，整份 PDF 的提取文本作为父文档。编号识别是启发式方法，需检查实际分块。

无 OCR：扫描页会记录警告，整份无文本时直接报错。多栏、表格、公式的阅读顺序取决于 pypdf，原 PDF 版式不会被完整恢复。`manifest.json` 中保存识别提示，应结合 `pages.jsonl` 与原页抽查。

## 输出

每次成功处理后打印独立批次目录：`rag-practice/prepared/<strategy>/<batch-id>/`。

| 文件 | 内容 |
| --- | --- |
| `pages.jsonl` | 逐页文本、物理页码、原文件哈希及字符偏移 |
| `children.jsonl` | 用于嵌入和检索的文本块，包含 `id`、`page_content`、`metadata` |
| `parents.jsonl` | 完整父章节；递归模式为空 |
| `manifest.json` | 处理参数、数量及警告；完成后最后写入 |

结构模式子块的 `metadata.parent_id` 指向父文档 `id`。所有块保留来源相对路径、文件哈希、物理页范围和文本偏移。字符偏移对应同一 PDF 各页以两个换行拼接后的文本，区间为 `[start, end)`。相同文件、路径与参数生成稳定块 ID；不同批次不得混用父子文件。

可用 `--input-dir` 和 `--output-dir` 指定其他目录。每次生成新批次，避免覆盖已有产物；历史批次需自行按需要清理。默认输出目录已加入 Git 忽略规则。

## 检索子块，回取完整父文档

以下代码放在与 `data_preparation.py` 同目录的脚本中。检索器的返回值需转成 `children.jsonl` 的记录格式，保留 `metadata.parent_id`。

```python
from pathlib import Path
from data_preparation import read_jsonl, build_parent_context

batch = Path("填入程序打印的结构分块批次目录")
parents = read_jsonl(batch / "parents.jsonl")
children = read_jsonl(batch / "children.jsonl")

# 将 children 的 page_content 嵌入并入库，同时保存其 id 和 metadata。
# 实际使用时，将下面演示记录替换为向量数据库返回的子块命中。
hits = children[:2]
context = build_parent_context(hits, parents, max_chars=30000)

# 把 context 作为证据上下文交给 LLM；本脚本本身不调用模型。
# 多个子块命中同一父文档时自动去重，顺序沿用首次命中顺序。
```

`max_chars` 是可选字符预算，不是 token 预算。超出时函数报错而不截断父文档；调用方应减少父文档数量、使用更长上下文的模型，或提高 `--parent-level` 后重新准备数据。传入的证据文本应作为数据处理，不执行其中的指令。

## 验证

```powershell
python -m unittest discover -s rag-practice -p test_data_preparation.py
```

测试覆盖同页书签、章节跨页、父子映射、上下文去重与预算、普通递归模式、无标题回退及非法参数。
