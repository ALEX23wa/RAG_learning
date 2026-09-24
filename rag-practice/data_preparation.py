"""PDF 文本提取与分块：python data_preparation.py --strategy structure。

children.jsonl 用于向量检索；结构模式下通过 parent_id 从 parents.jsonl
回取完整章节作为生成上下文。这里只准备数据，不调用嵌入模型或 LLM。
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import sys
from uuid import NAMESPACE_URL, uuid4, uuid5

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


BASE_DIR = Path(__file__).resolve().parent
SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", ". ", " ", ""]
NUMBERED = re.compile(r"^(\d+(?:\.\d+)+)\.?\s+\S")
CHINESE = re.compile(r"^第[一二三四五六七八九十百零〇\d]+([章节])\s*\S")


@dataclass(frozen=True)
class Heading:
    offset: int
    title: str
    level: int
    method: str


def stable_id(value: str) -> str:
    return str(uuid5(NAMESPACE_URL, value))


def heading_level(title: str, default: int) -> int:
    match = NUMBERED.match(title)
    if match:
        return len(match.group(1).split("."))
    match = CHINESE.match(title)
    if match:
        return 1 if match.group(1) == "章" else 2
    return default


def join_pages(pages: list[dict]) -> str:
    """记录字符偏移，保证分块能追溯到物理页（从 1 开始）。"""
    parts, offset = [], 0
    for page in pages:
        page["start"] = offset
        parts.append(page["text"])
        offset += len(page["text"])
        page["end"] = offset
        offset += 2
    return "\n\n".join(parts)


def extract_pdf(path: Path, root: Path) -> tuple[list[dict], str, list[Heading], list[str]]:
    reader = PdfReader(path)
    if reader.is_encrypted:
        raise ValueError(f"加密 PDF 请先解密：{path.name}")
    file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    source = path.relative_to(root).as_posix()
    doc_id = stable_id(f"{source}:{file_hash}")
    pages, warnings = [], []
    for index, page in enumerate(reader.pages, 1):
        text = (page.extract_text() or "").replace("\r\n", "\n").replace("\r", "\n")
        text = text.replace("\x00", "").strip()
        if not text:
            warnings.append(f"{source} 第 {index} 页无可提取文本，可能需要 OCR")
        pages.append({"doc_id": doc_id, "source": source, "file_sha256": file_hash,
                      "page": index, "text": text})
    text = join_pages(pages)
    if not text.strip():
        raise ValueError(f"PDF 没有可提取文本，请先 OCR：{source}")

    headings = []

    def visit(items, depth=1):
        for item in items:
            if isinstance(item, list):
                visit(item, depth + 1)
                continue
            title = " ".join(str(item.title).split())
            page_index = reader.get_destination_page_number(item)
            if page_index is None or not 0 <= page_index < len(pages) or not title:
                warnings.append(f"{source} 无法定位书签：{title}")
                continue
            page = pages[page_index]
            # 允许标题在 PDF 中断行、含制表符或不间断空格；保留原文偏移。
            pattern = r"\s*".join(re.escape(c) for c in title if not c.isspace())
            match = re.search(r"(?m)^[ \t]*" + pattern, page["text"], re.IGNORECASE)
            if match:
                headings.append(Heading(page["start"] + match.start(), title,
                                        heading_level(title, depth), "bookmark_text"))
            else:
                warnings.append(f"{source} 书签标题未匹配正文，未猜测边界：{title}（第 {page_index + 1} 页）")

    visit(reader.outline)
    if not headings:
        headings = detect_numbered_headings(text)
        warnings.append(f"{source} 无可用书签标题，改用正文编号标题识别")
    # 同一位置的重复书签只保留层级最高的一项。
    unique = {}
    for h in sorted(headings, key=lambda h: (h.offset, h.level)):
        unique.setdefault(h.offset, h)
    return pages, text, list(unique.values()), warnings


def detect_numbered_headings(text: str) -> list[Heading]:
    """无书签时识别 1.2、1.2.3、第一章等标题；不声称恢复任意 PDF 版式。"""
    headings = []
    for match in re.finditer(r"(?m)^\s*([^\n]+)", text):
        title = match.group(1).strip()
        if (len(title) <= 160 and not re.search(r"\.{3,}|…{2,}", title)
                and (NUMBERED.match(title) or CHINESE.match(title))):
            headings.append(Heading(match.start(1), title, heading_level(title, 1), "numbered_text"))
    return headings


def page_range(pages: list[dict], start: int, end: int) -> tuple[int, int]:
    touched = [p["page"] for p in pages if p["end"] > start and p["start"] < end]
    if not touched:
        raise ValueError("文本块无法定位到原始页")
    return touched[0], touched[-1]


def build_chunks(pages: list[dict], text: str, headings: list[Heading], *,
                 strategy: str = "structure", chunk_size: int = 800,
                 child_size: int = 300, overlap: int = 50,
                 parent_level: int = 2) -> tuple[list[dict], list[dict]]:
    size = child_size if strategy == "structure" else chunk_size
    if strategy not in {"recursive", "structure"}:
        raise ValueError("strategy 必须是 recursive 或 structure")
    if size <= 0 or not 0 <= overlap < size or parent_level < 1:
        raise ValueError("块大小须为正数，0 <= overlap < 块大小，parent_level >= 1")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=overlap, separators=SEPARATORS,
        add_start_index=True, strip_whitespace=False,
    )
    source = {k: pages[0][k] for k in ("doc_id", "source", "file_sha256")}
    config = f"v1:{strategy}:{size}:{overlap}:{parent_level}"
    headings = sorted(headings, key=lambda h: h.offset)
    paths, stack = [], []
    for h in headings:
        while stack and stack[-1].level >= h.level:
            stack.pop()
        stack.append(h)
        paths.append([entry.title for entry in stack])
    offsets = [h.offset for h in headings]

    # 指定层级的完整章节是父文档；更深层级的小节保留在该父章节内部。
    boundaries = [(0, "文档前言", [], "document")]
    if strategy == "structure":
        for i, h in enumerate(headings):
            if h.level <= parent_level:
                item = (h.offset, h.title, paths[i], h.method)
                if h.offset == 0:
                    boundaries[0] = item
                else:
                    boundaries.append(item)
    parents, children = [], []
    for index, (start, title, path, method) in enumerate(boundaries):
        end = boundaries[index + 1][0] if index + 1 < len(boundaries) else len(text)
        content = text[start:end]
        if not content.strip():
            continue
        parent_id = stable_id(f"{source['doc_id']}:{config}:parent:{start}:{end}")
        if not headings:
            title, method = Path(source["source"]).stem, "whole_document_fallback"
        if strategy == "structure":
            first, last = page_range(pages, start, end)
            parents.append({"id": parent_id, "page_content": content, "metadata": {
                **source, "parent_id": parent_id, "title": title, "heading_path": path,
                "heading_method": method, "page_start": first, "page_end": last,
                "start": start, "end": end, "strategy": strategy,
            }})
        for document in splitter.create_documents([content]):
            if not document.page_content.strip():
                continue
            begin = start + document.metadata["start_index"]
            finish = begin + len(document.page_content)
            if begin < start or text[begin:finish] != document.page_content:
                raise ValueError("分块偏移与原始文本不一致")
            first, last = page_range(pages, begin, finish)
            heading_index = bisect_right(offsets, begin) - 1
            metadata = {**source, "strategy": strategy, "page_start": first,
                        "page_end": last, "start": begin, "end": finish,
                        "heading_path": paths[heading_index] if heading_index >= 0 else []}
            if strategy == "structure":
                metadata["parent_id"] = parent_id
            children.append({"id": stable_id(f"{source['doc_id']}:{config}:child:{begin}:{finish}"),
                             "page_content": document.page_content, "metadata": metadata})
    return parents, children


def resolve_parents(hits: list[dict], parents: list[dict]) -> list[dict]:
    """输入 children.jsonl 格式的检索命中，按命中顺序回取父文档并去重。"""
    lookup = {parent["id"]: parent for parent in parents}
    selected, seen = [], set()
    for hit in hits:
        parent_id = hit["metadata"].get("parent_id")
        if parent_id not in lookup:
            raise ValueError(f"未找到父文档 {parent_id}；请使用同一批结构分块产物")
        if parent_id not in seen:
            selected.append(lookup[parent_id])
            seen.add(parent_id)
    return selected


def build_parent_context(hits: list[dict], parents: list[dict], *, max_chars: int | None = None) -> str:
    """将完整父章节交给调用方作为 LLM 上下文；超过预算时报错，绝不静默截断。"""
    selected = resolve_parents(hits, parents)
    context = "\n\n".join(
        f"[P{i}] {p['metadata']['source']} | {p['metadata']['title']} | "
        f"物理页 {p['metadata']['page_start']}-{p['metadata']['page_end']}\n{p['page_content']}"
        for i, p in enumerate(selected, 1)
    )
    if max_chars is not None and len(context) > max_chars:
        raise ValueError("完整父文档超出上下文字符预算；减少命中父文档数或提高 parent_level 后重新分块")
    return context


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def prepare(input_dir: Path, output_dir: Path, **options) -> Path:
    root = input_dir.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("输入路径必须是目录")
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf")
    if not files:
        raise ValueError(f"未找到 PDF：{root}")
    all_pages, all_parents, all_children, warnings = [], [], [], []
    for file in files:
        print(f"处理：{file.relative_to(root)}", flush=True)
        pages, text, headings, notes = extract_pdf(file, root)
        parents, children = build_chunks(pages, text, headings, **options)
        if options.get("strategy", "structure") == "structure" and not headings:
            notes.append(f"{file.name} 未识别标题，使用完整 PDF 文本作为父文档")
        all_pages.extend(pages)
        all_parents.extend(parents)
        all_children.extend(children)
        warnings.extend(notes)
    # 每次输出独立批次，防止不同 PDF 版本的父子块混用；manifest 最后写入。
    output = output_dir.resolve() / options.get("strategy", "structure") / uuid4().hex
    output.mkdir(parents=True)
    for name, records in (("pages", all_pages), ("parents", all_parents), ("children", all_children)):
        with (output / f"{name}.jsonl").open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    manifest = {"schema_version": 1, "input_dir": str(root), "options": options,
                "pdf_count": len(files), "page_count": len(all_pages),
                "parent_count": len(all_parents), "child_count": len(all_children),
                "warnings": warnings}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"完成：PDF={len(files)} pages={len(all_pages)} parents={len(all_parents)} children={len(all_children)}")
    print(f"输出：{output}")
    if warnings:
        print(f"有 {len(warnings)} 条提取/标题识别提示，详见 manifest.json")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=BASE_DIR / "papers")
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "prepared")
    parser.add_argument("--strategy", choices=["recursive", "structure"], default="structure")
    parser.add_argument("--chunk-size", type=int, default=800, help="递归模式块大小，单位为字符")
    parser.add_argument("--child-size", type=int, default=300, help="结构模式检索子块大小，单位为字符")
    parser.add_argument("--overlap", type=int, default=50, help="重叠字符数，须小于所选模式块大小")
    parser.add_argument("--parent-level", type=int, default=2, help="父章节边界最大层级；例如 3.2 为 2，3.2.1 为 3")
    args = vars(parser.parse_args())
    size = args["child_size"] if args["strategy"] == "structure" else args["chunk_size"]
    if size <= 0 or not 0 <= args["overlap"] < size or args["parent_level"] < 1:
        parser.error("块大小须为正数，0 <= overlap < 块大小，parent_level >= 1")
    prepare(**args)


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    try:
        main()
    except (ValueError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
