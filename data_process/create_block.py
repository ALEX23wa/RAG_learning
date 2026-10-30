"""
Markdown 父子分块：子块用于向量召回，父块用于补全回答上下文。
支持本地 Markdown 图片编码；结构块包括标题、图片、代码和管道表格。
"""

import os
import re
import warnings
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5
from urllib.parse import unquote, urlsplit

from markdown_it import MarkdownIt

# 必须在导入 MinerU / Hugging Face 前设置，避免模型和下载缓存写入 C 盘。
# huggingface_hub 在 import 阶段就固定了 HF_HOME，所以这段必须排在第三方导入之前。
MINERU_HOME = Path(os.environ.get("MINERU_HOME", r"D:\Program Files\MinerU")).expanduser().resolve()
os.environ.setdefault("MINERU_HOME", str(MINERU_HOME))
os.environ.setdefault("MINERU_MODEL_BASE_DIR", str(MINERU_HOME / "models"))
os.environ.setdefault("HF_HOME", str(MINERU_HOME / "cache" / "huggingface"))
os.environ.setdefault("HF_HUB_CACHE", str(Path(os.environ["HF_HOME"]) / "hub"))
os.environ.setdefault("HF_XET_CACHE", str(Path(os.environ["HF_HOME"]) / "xet"))
os.environ.setdefault("MODELSCOPE_CACHE", str(MINERU_HOME / "cache" / "modelscope"))
# 模型都已下载到本地，禁止再联网做版本校验或下载。需要联网时先显式设置 HF_HUB_OFFLINE=0。
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
# 使用 ModelScope 补齐公式等模型，避免 Hugging Face 连接超时。
# 如需其他下载源，可在运行前显式设置 MINERU_MODEL_SOURCE。
os.environ.setdefault("MINERU_MODEL_SOURCE", "modelscope")

# eva_clip 仍从 timm.models.layers 导入，timm 已弃用该路径，无法从上游消除。
warnings.filterwarnings(
    "ignore",
    message=r"Importing from timm\.models\.layers is deprecated",
    category=FutureWarning,
)
# Windows 默认不支持符号链接，huggingface_hub 每次命中缓存都会提示。
warnings.filterwarnings(
    "ignore",
    message=r".*cache-system uses symlinks by default",
    category=UserWarning,
)

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter


def _image_tokens(tokens):
    """只读取 Markdown 图片节点，忽略代码、转义标记和普通链接。"""
    for token in tokens:
        if token.type == "image":
            yield token
        elif token.children:
            yield from _image_tokens(token.children)


def _image_text(tokens) -> str:
    """提取正文和图片说明，不把图片路径或标题作为正文送给模型。"""
    parts = []
    for token in tokens:
        if token.children:
            parts.append(_image_text(token.children))
        elif token.type in {"text", "code_inline"}:
            parts.append(token.content)
        elif token.type in {"softbreak", "hardbreak"} or token.block:
            parts.append("\n")
    return "".join(parts).strip()


class VisualizedBGEEmbeddings(Embeddings):
    """文本用文本编码；本地 Markdown 图片用图像或图文编码，多图归一化平均。"""

    def __init__(self, model, image_base_dir: str | Path | None = None, batch_size: int = 32):
        if batch_size <= 0:
            raise ValueError("batch_size 必须大于 0")
        self.model = model
        self.model.eval()
        self.batch_size = batch_size
        self.image_base_dir = Path(image_base_dir or Path.cwd()).expanduser().resolve()
        self._markdown_env: dict = {}

    def for_markdown(self, path: Path, text: str):
        """为一次建库绑定来源和引用定义，不修改可复用的原适配器。"""
        scoped = copy(self)
        scoped.image_base_dir = path.parent
        scoped._markdown_env = {}
        MarkdownIt().parse(text, scoped._markdown_env)
        return scoped

    def _image_path(self, destination: str) -> Path:
        # Windows 盘符不能按 URL scheme 判断。
        if re.match(r"^[A-Za-z]:[\\/]", destination):
            path = Path(unquote(destination))
        else:
            url = urlsplit(destination)
            if url.scheme or url.netloc:
                raise ValueError(f"仅支持本地 Markdown 图片，收到：{destination}")
            path = Path(unquote(url.path)).expanduser()
        if not path.is_absolute():
            path = self.image_base_dir / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Markdown 图片不存在：{path}（原路径：{destination}）")
        return path

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        import torch
        import torch.nn.functional as F

        vectors: list[list[float]] = []
        parser = MarkdownIt()
        with torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                for text in texts[start:start + self.batch_size]:
                    tokens = parser.parse(text, dict(self._markdown_env))
                    images = list(_image_tokens(tokens))
                    if images:
                        clean_text = _image_text(tokens)
                        image_vectors = []
                        for image in images:
                            image_path = self._image_path(image.attrGet("src"))
                            kwargs = {"image": str(image_path)}
                            if clean_text:
                                kwargs["text"] = clean_text
                            image_vectors.append(self.model.encode(**kwargs).float())
                        encoded = F.normalize(torch.stack(image_vectors).mean(dim=0), p=2, dim=-1)
                    else:
                        encoded = self.model.encode(text=text)
                    vectors.append(encoded.float().cpu().tolist()[0])
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def _table_cells(line: str) -> list[str]:
    """只把未转义的竖线视为列边界，支持省略两侧竖线的表格。"""
    cells: list[str] = []
    start = 0
    for index, char in enumerate(line):
        if char != "|":
            continue
        backslashes = len(line[:index]) - len(line[:index].rstrip("\\"))
        if backslashes % 2 == 0:
            cells.append(line[start:index].strip())
            start = index + 1
    cells.append(line[start:].strip())
    if line.strip().startswith("|"):
        cells.pop(0)
    if line.rstrip().endswith("|") and cells and cells[-1] == "":
        cells.pop()
    return cells


def _markdown_blocks(text: str) -> list[Document]:
    """扫描标题、围栏代码、缩进代码和管道表格，保留结构块的原始空白。"""
    text = text.lstrip("\ufeff")
    lines = text.splitlines()
    # inline 节点的行范围包含完整段落，支持行内图片、多图及跨行图片标记。
    image_ranges = {
        token.map[0]: token.map[1]
        for token in MarkdownIt().parse(text)
        if token.type == "inline" and token.map and list(_image_tokens([token]))
    }
    blocks: list[Document] = []
    headers: dict[str, str] = {}
    pending: list[str] = []
    section_index = 0

    def emit(content: list[str], kind: str, **metadata) -> None:
        if any(line.strip() for line in content):
            blocks.append(Document(
                page_content="\n".join(content),
                metadata={**headers, "section_index": section_index,
                          "content_type": kind, **metadata},
            ))

    def flush() -> None:
        emit(pending, "text")
        pending.clear()

    index = 0
    while index < len(lines):
        line = lines[index]
        fence = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line)
        if fence and not (fence[1][0] == "`" and "`" in fence[2]):
            flush()
            marker, info = fence.groups()
            content = [line]
            index += 1
            closing = re.compile(r"^ {0,3}" + re.escape(marker[0]) + "{" + str(len(marker)) + r",}[ \t]*$")
            while index < len(lines) and not closing.match(lines[index]):
                content.append(lines[index])
                index += 1
            unclosed = index == len(lines)
            content.append(marker if unclosed else lines[index])
            if not unclosed:
                index += 1
            emit(content, "code", code_style="fenced", language=info.strip(),
                 fence_repaired=unclosed)
            continue

        # 缩进代码仅在块边界开始，避免将普通段落的续行误判为代码。
        if (line.startswith("    ") or line.startswith("\t")) and (
            index == 0 or not lines[index - 1].strip()
        ):
            flush()
            content = []
            while index < len(lines) and (
                not lines[index].strip() or lines[index].startswith(("    ", "\t"))
            ):
                content.append(lines[index])
                index += 1
            emit(content, "code", code_style="indented", language="")
            continue

        # 含图片的完整段落作为原子块，路径和引用标记不会被长度切分截断。
        if index in image_ranges:
            flush()
            end = image_ranges[index]
            emit(lines[index:end], "image")
            index = end
            continue

        # Markdown 表格必须至少两行，且第二行是分隔行，才能被识别为表格。
        if index + 1 < len(lines) and "|" in line:
            cells = _table_cells(line)
            delimiters = _table_cells(lines[index + 1])
            if cells and len(cells) == len(delimiters) and all(
                re.fullmatch(r":?-{3,}:?", cell) for cell in delimiters
            ):
                flush()
                content = lines[index:index + 2]
                index += 2
                while index < len(lines) and "|" in lines[index] and len(_table_cells(lines[index])) == len(cells):
                    content.append(lines[index])
                    index += 1
                emit(content, "table")
                continue

        heading = re.match(r"^ {0,3}(#{1,6})(?:[ \t]+(.*)|$)", line)
        if heading:
            flush()
            level = len(heading[1])
            headers = {key: value for key, value in headers.items() if int(key.split()[1]) < level}
            title = re.sub(r"[ \t]+#+[ \t]*$", "", heading[2] or "").strip()
            headers[f"Header {level}"] = title
            section_index += 1
        pending.append(line)
        index += 1
    flush()
    return blocks


def _split_block(document: Document, size: int, overlap: int = 0) -> list[Document]:
    """结构块按整行拆分并重复结构头；单行超长时保留内容并标记。"""
    
    kind = document.metadata["content_type"]
    if kind == "image":
        return [
            Document(
                page_content=document.page_content,
                metadata={
                    **document.metadata,  # 表示把字典里的键值对展开到当前字典中
                    "oversized": len(document.page_content) > size,
                },
            )
        ]
    if kind == "text":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=size, chunk_overlap=overlap,
            separators=["\n\n", "\n", "。", "！", "？", ". ", "! ", "? ", "；", "; ", " ", ""],
        )
        parts = splitter.split_text(document.page_content)
    else:
        lines = document.page_content.split("\n")
        prefix, suffix = [], []
        if kind == "table":
            prefix, lines = lines[:2], lines[2:]
        elif document.metadata.get("code_style") == "fenced":
            prefix, lines, suffix = lines[:1], lines[1:-1], lines[-1:]

        def render(rows: list[str]) -> str:
            return "\n".join(prefix + rows + suffix)

        parts = []
        batch: list[str] = []
        for line in lines:
            if batch and len(render(batch + [line])) > size:
                parts.append(render(batch))
                batch = []
            batch.append(line)
        if batch or not parts:
            parts.append(render(batch))
    return [Document(page_content=part, metadata={
        **document.metadata, "oversized": len(part) > size,
    }) for part in parts]


@dataclass
class ParentChildIndex:
    """向量库只保存子块；父块通过 parent_id 保存在内存字典中。"""

    vector_store: InMemoryVectorStore
    parent_documents: dict[str, Document]
    child_documents: list[Document]

    def search_parents(self, query: str, k: int = 4) -> list[Document]:
        """召回 k 个子块，按首次命中顺序返回去重父块（数量可能小于 k）。"""
        if k <= 0:
            raise ValueError("k 必须大于 0")
        if not self.child_documents:
            return []
        children = self.vector_store.similarity_search(query, k=k)
        parent_ids = dict.fromkeys(child.metadata["parent_id"] for child in children)
        return [self.parent_documents[parent_id] for parent_id in parent_ids]


def split_markdown(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    *,
    parent_chunk_size: int = 2000,
    source: str = "",
) -> tuple[list[Document], list[Document]]:
    """纯文本分块，不调用 Embedding；返回 (父块列表, 子块列表)。

    按 # 至 ###### 标题划分章节，保留标题正文和标题层级元数据。
    超长章节继续按 parent_chunk_size 拆成无重叠父块，再在每个父块内
    按 chunk_size / chunk_overlap 切出子块，子块不会跨越父块边界。
    长度单位均为字符；重叠量是上限，实际重叠取决于段落和句子边界。
    无标题文本同样可分块，空白文本返回两个空列表。

    围栏代码（反引号或波浪线）、缩进代码和 Markdown 管道表格独立成块。
    代码保留缩进；超长代码按行拆分并重复围栏及语言标记；超长表格按行
    拆分并重复表头和分隔行。结构块不做重叠，避免重复数据行。
    若单行连同结构头仍超长，保留整行并标记 metadata['oversized']=True。
    围栏未闭合时自动补齐，标记 fence_repaired=True。content_type 区分
    text/image/code/table；含图片段落整体保留，超长时标记 oversized。
    支持内联及引用式 Markdown 图片；HTML img 标签暂不作为图片处理。
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap 必须满足 0 <= chunk_overlap < chunk_size")
    if parent_chunk_size < chunk_size:
        raise ValueError("parent_chunk_size 必须大于或等于 chunk_size")

    # 来源、内容和配置相同则 ID 稳定；不同文件中的同名章节不会混淆。
    document_id = uuid5(NAMESPACE_URL, f"{source}\n{text}")
    config = f"structured-v3:{parent_chunk_size}:{chunk_size}:{chunk_overlap}"
    parents: list[Document] = []
    children: list[Document] = []
    for block in _markdown_blocks(text):
        for parent in _split_block(block, parent_chunk_size):
            parent_index = len(parents)
            parent_id = str(uuid5(document_id, f"{config}:parent:{parent_index}"))
            parent.id = parent_id
            parent.metadata.update(
                source=source, parent_id=parent_id, chunk_type="parent",
                parent_index=parent_index,
            )
            parents.append(parent)
            for child_index, child in enumerate(_split_block(parent, chunk_size, chunk_overlap)):
                child_id = str(uuid5(document_id, f"{config}:child:{parent_index}:{child_index}"))
                child.id = child_id
                child.metadata.update(
                    child_id=child_id, chunk_type="child", child_index=child_index,
                )
                children.append(child)
    return parents, children


def create_block(
    path: str | Path,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    *,
    parent_chunk_size: int = 2000,
    embeddings: Embeddings | None = None,
) -> ParentChildIndex:
    """读取 UTF-8 Markdown，生成父子块并仅将子块写入内存向量库。

    embeddings 可传入 VisualizedBGEEmbeddings 等适配器或测试模型；省略时加载本地
    Visualized-BGE。图片路径相对于 Markdown 文件目录解析，不下载远程图片。
    只想查看分块时直接调用 split_markdown。
    返回值包含 vector_store、parent_documents 和 child_documents，
    可通过 result.search_parents(query) 检索完整父块。所有数据仅驻留内存。

    Example:
        result = create_block(Path("papers/example.md"), embeddings=my_embeddings)
        parents = result.search_parents("这篇论文提出了什么方法？", k=4)
    """
    path = Path(path).expanduser().resolve()
    text = path.read_text(encoding="utf-8-sig")
    parents, children = split_markdown(
        text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        parent_chunk_size=parent_chunk_size,
        source=str(path),
    )
    if embeddings is None:
        from visual_bge.modeling import Visualized_BGE

        embeddings = VisualizedBGEEmbeddings(
            model=Visualized_BGE(model_name_bge="BAAI/bge-m3",model_weight=str(Path(__file__).resolve().parent.parent / "Model" / "Visualized_m3.pth")),
            batch_size=32,
        )
    if isinstance(embeddings, VisualizedBGEEmbeddings):
        embeddings = embeddings.for_markdown(path, text)
    vector_store = InMemoryVectorStore(embedding=embeddings)
    if children:
        vector_store.add_documents(children)
    return ParentChildIndex(
        vector_store=vector_store,
        parent_documents={parent.id: parent for parent in parents},
        child_documents=children,
    )

if __name__ == "__main__":
    from visual_bge.modeling import Visualized_BGE

    result = create_block(
        path=Path(__file__).resolve().parent.parent / "papers" / "extracted" / "rag.md",
        embeddings=VisualizedBGEEmbeddings(
            model=Visualized_BGE(
                model_name_bge="BAAI/bge-m3",
                model_weight=str(Path(__file__).resolve().parent.parent / "Model" / "Visualized_m3.pth"),
            ),
            batch_size=32,
        ),
    )
    for parent in result.search_parents("什么是 RAG？", k=4):
        print(parent.page_content)
        print("-" * 60)

    print(f"父块数量: {len(result.parent_documents)}")
    print(f"子块数量: {len(result.child_documents)}")