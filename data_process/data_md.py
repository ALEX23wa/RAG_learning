import os
from pathlib import Path
from typing import Literal
from pypdf import PdfReader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_DIR = PROJECT_ROOT / "papers"
OUTPUT_DIR = PROJECT_DIR / "extracted"

# 必须在导入 MinerU / Hugging Face 前设置，避免模型和下载缓存写入 C 盘。
MINERU_HOME = Path(r"D:\Program Files\MinerU")
os.environ["MINERU_HOME"] = str(MINERU_HOME)
os.environ["MINERU_MODEL_BASE_DIR"] = str(MINERU_HOME / "models")
os.environ["HF_HOME"] = str(MINERU_HOME / "cache" / "huggingface")
os.environ["HF_HUB_CACHE"] = str(MINERU_HOME / "cache" / "huggingface" / "hub")
os.environ["HF_XET_CACHE"] = str(MINERU_HOME / "cache" / "huggingface" / "xet")
os.environ["MODELSCOPE_CACHE"] = str(MINERU_HOME / "cache" / "modelscope")
# 本机 GT 730 仅有 1 GB 显存；隐藏 Vulkan 设备，让 llama.cpp 使用 CPU。
# Windows 的 C 运行库会将空字符串视为删除变量；空格表示存在但设备列表为空。
os.environ["GGML_VK_VISIBLE_DEVICES"] = " "
os.environ["MINERU_MODEL_SMALL_BACKEND"] = "onnx"
os.environ["MINERU_MODEL_VLM_ENGINE"] = "llama-cpp"

# 第一步先处理文本型PDF
def extract_text_from_pdf(pdf_path: Path | str) -> list[dict]:
    """
    Extract text from a PDF file using pypdf, but can't process images.
  
    Args:
        pdf_path (Path | str): The path to the PDF file.
        """
    pdf_path = Path(pdf_path).resolve()
    reader = PdfReader(pdf_path)
    if reader.is_encrypted:
        raise ValueError(f"The PDF file {pdf_path} is encrypted and cannot be processed.")

    pages=[]
    for page_number,page in enumerate(reader.pages,start=1):
        text = (page.extract_text() or "").strip()
        pages.append({"source": str(pdf_path.relative_to(PROJECT_ROOT)) if pdf_path.is_relative_to(PROJECT_ROOT) else str(pdf_path),
            "page": page_number,
            "text": text})

    return pages

# 第一步的进阶方法，使用unstructured OCR技术处理文本型PDF和图片型PDF(但似乎效果很一般) 可以试试MinerU
def extract_text_from_pdf_ocr(pdf_path:Path | str) -> list[dict]:
    """
    提取 PDF 各页文字；对扫描页使用 OCR技术进行识别, 返回每页的内容。

    Args:
        pdf_path (Path | str): The path to the PDF file.
    """
    from unstructured.partition.pdf import partition_pdf

    pdf_path = Path(pdf_path)

    try :
        # 使用 unstructured 库的 partition_pdf 方法提取 PDF 内容
        elements = partition_pdf(
            filename=str(pdf_path),
            strategy="ocr_only",  # 使用 OCR 技术处理扫描页
            languages=["eng", "chi_sim"],  # 支持英文和简体中文
        )
    except Exception as exc:
        raise ValueError(f"处理PDF失败:{pdf_path}") from exc

    pages: dict[int, list[str]] = {}
    for element in elements:
        page = element.metadata.page_number
        if page is not None and element.text.strip():
            pages.setdefault(page, []).append(element.text.strip())

    return [
        {
            "source": str(pdf_path),
            "page": page,
            "text": "\n".join(texts),
        }
        for page, texts in sorted(pages.items())
    ]


def pages_to_markdown(pages: list[dict]) -> str:
    """将 PDF 逐页提取结果转成带页码的 Markdown。"""
    sections = []
    for page in pages:
        text = page["text"].strip()
        if text:
            sections.append(f'## 第 {page["page"]} 页\n\n{text}')
    return "\n\n".join(sections) + ("\n" if sections else "")


def pdf_to_markdown(pdf_path: Path | str, use_ocr: bool = False) -> Path:
    """提取 PDF 并将 Markdown 写入 papers/extracted 目录。"""
    pdf_path = Path(pdf_path)
    pages = (
        extract_text_from_pdf_ocr(pdf_path)
        if use_ocr
        else extract_text_from_pdf(pdf_path)
    )
    markdown = pages_to_markdown(pages)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{pdf_path.stem}.md"
    output_path.write_text(markdown, encoding="utf-8")
    return output_path

# MinerU 4.x 提供了更强的 PDF 解析能力，支持图片、表格、公式等结构化内容的提取
def pdf_to_markdown_mineru(
    pdf_path: Path | str,
    output_dir: Path | str | None = None,
    *,
    tier: Literal["flash", "basic", "standard", "advanced"] = "standard",
    ocr_mode: Literal["auto", "txt", "ocr"] = "auto",
    page_range: str = "",
) -> Path:
    """使用 MinerU 4.x 将 PDF 转为 Markdown，并保存图片和结构化 JSON

    默认输出：papers/extracted/mineru/markdown.md。
    output_dir 指定本次结果目录；重复运行会覆盖该目录中的同名结果文件。
    ocr_mode="auto" 自动选择文字提取方式，扫描件可指定 "ocr"。
    page_range 默认处理全部页面；"1-3" 表示仅处理第 1 至 3 页
    首次使用模型可能下载权重；flash + txt 仅提取已有文字层

    Example:
        md_path = pdf_to_markdown_mineru(PROJECT_DIR / "rag.pdf", page_range="1-3")
        markdown = md_path.read_text(encoding="utf-8")
    """
    pdf_path = Path(pdf_path).expanduser().resolve()
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF 文件不存在: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"请输入 PDF 文件: {pdf_path}")

    # 延迟导入，其他提取方法不依赖 MinerU。
    from mineru import parse
    from mineru.parser.writer import FileBasedDataWriter

    result = parse(
        pdf_path,
        tier=tier,
        ocr_mode=ocr_mode,
        page_range=page_range,
    )
    destination = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else OUTPUT_DIR / "mineru"
    )
    destination.mkdir(parents=True, exist_ok=True)
    result.save(FileBasedDataWriter(str(destination)))
    return destination / "markdown.md"

if __name__ == "__main__":
    print(pdf_to_markdown_mineru(PROJECT_DIR / "rag.pdf"))
