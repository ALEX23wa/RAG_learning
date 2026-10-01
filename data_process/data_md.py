import os
import argparse
import importlib.util
import logging
import re
import subprocess
import sys
from pathlib import Path
from typing import Literal
from pypdf import PdfReader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_DIR = PROJECT_ROOT / "papers"
OUTPUT_DIR = PROJECT_DIR / "extracted"

# 必须在导入 MinerU / Hugging Face 前设置，避免模型和下载缓存写入 C 盘。
MINERU_HOME = Path(os.environ.get("MINERU_HOME", r"D:\Program Files\MinerU")).expanduser().resolve()
os.environ.setdefault("MINERU_HOME", str(MINERU_HOME))
os.environ.setdefault("MINERU_MODEL_BASE_DIR", str(MINERU_HOME / "models"))
os.environ.setdefault("HF_HOME", str(MINERU_HOME / "cache" / "huggingface"))
os.environ.setdefault("HF_HUB_CACHE", str(Path(os.environ["HF_HOME"]) / "hub"))
os.environ.setdefault("HF_XET_CACHE", str(Path(os.environ["HF_HOME"]) / "xet"))
os.environ.setdefault("MODELSCOPE_CACHE", str(MINERU_HOME / "cache" / "modelscope"))

logger = logging.getLogger(__name__)


def _mineru_environment(device: str) -> dict[str, str]:
    """每次解析独立设置后端，防止 native GPU 缓存妨碍 CPU 回退。"""
    env = os.environ.copy()
    env.update(MINERU_MODEL_SMALL_BACKEND="onnx", MINERU_MODEL_VLM_ENGINE="llama-cpp",
               MINERU_DEVICE_MODE="cpu", MINERU_TABLE_DEVICE="cpu", PYTHONUTF8="1")
    if device == "cpu":
        # Windows CRT 会删除空字符串变量；空格代表 Vulkan 设备列表为空。
        env["GGML_VK_VISIBLE_DEVICES"] = " "
        env["CUDA_VISIBLE_DEVICES"] = "-1"
    return env


def mineru_gpu_devices() -> list[str]:
    """用安装包自带的 llama.cpp 实测设备；不能用 torch.cuda 判断 AMD Vulkan。"""
    spec = importlib.util.find_spec("mineru_llama_cpp")
    if spec is None or not spec.origin:
        raise ImportError("请先安装 environment.yml 中的 mineru-llama-cpp")
    executable = Path(spec.origin).parent / "bin" / ("llama-server.exe" if os.name == "nt" else "llama-server")
    try:
        probe = subprocess.run(
            [str(executable), "--list-devices"], capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=30, env=_mineru_environment("gpu"),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("GPU 探测失败，将使用 CPU: %s", exc)
        return []
    if probe.returncode:
        logger.warning("GPU 探测失败，将使用 CPU: %s", probe.stderr[-2000:])
        return []
    return re.findall(r"^\s*((?:Vulkan|CUDA|ROCm|Metal)\d*: .+)$", probe.stdout, re.MULTILINE)


def _gpu_failure(returncode: int, log: str) -> bool:
    """仅 GPU/内存/native 崩溃触发回退；模型下载和输入错误直接报告。"""
    markers = ("out of memory", "erroroutofdevicememory", "errordevicelost",
               "vk_error", "device lost", "failed to allocate", "vulkan error",
               "failed to create vulkan", "no vulkan devices", "cuda error")
    return any(marker in log.lower() for marker in markers) or returncode in (
        -6, -11, -1073741819, 3221225477, -1073740791, 3221226505,
    )

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
    device: Literal["auto", "gpu", "cpu"] = "auto",
) -> Path:
    """使用 MinerU 4.x 将 PDF 转为 Markdown，并保存图片和结构化 JSON

    默认输出：papers/extracted/mineru/markdown.md。
    output_dir 指定本次结果目录；重复运行会覆盖该目录中的同名结果文件。
    ocr_mode="auto" 自动选择文字提取方式，扫描件可指定 "ocr"。
    page_range 默认处理全部页面；"1-3" 表示仅处理第 1 至 3 页
    首次使用模型可能下载权重；flash + txt 仅提取已有文字层
    device="auto" 优先 Vulkan GPU，设备不可用或 GPU 运行失败时重试 CPU。
    device="gpu" 强制 GPU（失败报错），"cpu" 强制 CPU；ONNX 小模型始终使用 CPU。
    解析在独立子进程运行；详细日志为输出目录中的 mineru-gpu.log / mineru-cpu.log。

    Example:
        md_path = pdf_to_markdown_mineru(PROJECT_DIR / "rag.pdf", page_range="1-3")
        markdown = md_path.read_text(encoding="utf-8")
    """
    pdf_path = Path(pdf_path).expanduser().resolve()
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF 文件不存在: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"请输入 PDF 文件: {pdf_path}")

    if device not in {"auto", "gpu", "cpu"}:
        raise ValueError(f"未知设备: {device}")
    destination = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else OUTPUT_DIR / "mineru"
    )
    destination.mkdir(parents=True, exist_ok=True)
    devices = mineru_gpu_devices() if device != "cpu" else []
    if device == "gpu" and not devices:
        raise RuntimeError("MinerU llama.cpp 未检测到可用 GPU；可设置 device='auto' 自动回退 CPU")
    selected = "gpu" if devices else "cpu"
    logger.info("MinerU VLM 使用 %s；设备: %s；小模型使用 ONNX CPU", selected, devices)
    attempts = [selected, "cpu"] if selected == "gpu" and device == "auto" else [selected]
    for selected in attempts:
        log_path = destination / f"mineru-{selected}.log"
        command = [sys.executable, str(Path(__file__).resolve()), str(pdf_path),
                   "--output-dir", str(destination), "--tier", tier, "--ocr-mode", ocr_mode,
                   "--page-range", page_range, "--device", selected, "--worker"]
        with log_path.open("w", encoding="utf-8") as log_file:
            completed = subprocess.run(command, env=_mineru_environment(selected),
                                       stdout=log_file, stderr=subprocess.STDOUT)
        if completed.returncode == 0:
            output = destination / "markdown.md"
            if not output.is_file():
                raise RuntimeError(f"MinerU 未生成 Markdown；请查看 {log_path}")
            return output
        log = log_path.read_text(encoding="utf-8", errors="replace")
        if selected == "gpu" and device == "auto" and _gpu_failure(completed.returncode, log):
            logger.warning("GPU 解析失败，改用独立 CPU 进程重试；日志: %s", log_path)
            continue
        raise RuntimeError(f"MinerU {selected} 解析失败（退出码 {completed.returncode}）；"
                           f"日志: {log_path}\n{log[-4000:]}")
    raise RuntimeError("MinerU 未完成解析")


def _main() -> None:
    cli = argparse.ArgumentParser(description="MinerU PDF 转 Markdown，默认 GPU 优先、CPU 回退")
    cli.add_argument("pdf", nargs="?", type=Path, default=PROJECT_DIR / "rag.pdf")
    cli.add_argument("--output-dir", type=Path, default=OUTPUT_DIR / "mineru")
    cli.add_argument("--tier", choices=["flash", "basic", "standard", "advanced"], default="standard")
    cli.add_argument("--ocr-mode", choices=["auto", "txt", "ocr"], default="auto")
    cli.add_argument("--page-range", default="")
    cli.add_argument("--device", choices=["auto", "gpu", "cpu"], default="auto")
    cli.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = cli.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if args.worker:
        # 在任何 MinerU/native 导入前选定设备；父进程不加载模型。
        os.environ.update(_mineru_environment(args.device))
        from mineru import parse
        from mineru.config import VlmConfig
        from mineru.parser.writer import FileBasedDataWriter

        result = parse(args.pdf, tier=args.tier, ocr_mode=args.ocr_mode,
                       page_range=args.page_range, vlm_config=VlmConfig(engine="llama-cpp"))
        result.save(FileBasedDataWriter(str(args.output_dir)))
    else:
        print(pdf_to_markdown_mineru(args.pdf, args.output_dir, tier=args.tier,
                                    ocr_mode=args.ocr_mode, page_range=args.page_range, device=args.device))

if __name__ == "__main__":
    _main()
