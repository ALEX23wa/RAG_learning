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
    """构造独立 MinerU 子进程的后端环境，支持干净的 CPU 回退。

    Args:
        device: str：调用方已校验的 gpu 或 cpu；cpu 显式隐藏 Vulkan/CUDA 设备。

    Returns:
        dict[str, str]：当前进程环境的副本，加上 MinerU/设备/UTF-8 配置。

    Notes:
        不直接修改父进程环境；小型模型与表格组件固定使用 ONNX CPU，VLM 使用 llama.cpp。
    """
    env = os.environ.copy()
    env.update(MINERU_MODEL_SMALL_BACKEND="onnx", MINERU_MODEL_VLM_ENGINE="llama-cpp",
               MINERU_DEVICE_MODE="cpu", MINERU_TABLE_DEVICE="cpu", PYTHONUTF8="1")
    if device == "cpu":
        # Windows CRT 会删除空字符串变量；空格代表 Vulkan 设备列表为空。
        env["GGML_VK_VISIBLE_DEVICES"] = " "
        env["CUDA_VISIBLE_DEVICES"] = "-1"
    return env


def mineru_gpu_devices() -> list[str]:
    """调用 MinerU 自带的 llama.cpp 程序实测可用加速设备。

    Args:
        无显式输入参数。

    Returns:
        list[str]：Vulkan/CUDA/ROCm/Metal 设备描述；探测失败或未发现设备时返回空列表。

    Raises:
        ImportError：未安装或无法定位 mineru_llama_cpp 包。

    Notes:
        无显式参数；设备探测超时 30 秒，失败原因写日志。AMD Vulkan 不能通过 torch.cuda.is_available() 判断。
    """
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
    """判断子进程失败是否适合从 GPU 回退到 CPU。

    Args:
        returncode: int：MinerU 子进程退出码。
        log: str：该次运行的日志内容。

    Returns:
        bool：出现设备/内存错误标记或已知 native 崩溃码时为 True。

    Notes:
        模型下载失败或输入格式错误不因此被掩盖；匹配时忽略日志大小写。
    """
    markers = ("out of memory", "erroroutofdevicememory", "errordevicelost",
               "vk_error", "device lost", "failed to allocate", "vulkan error",
               "failed to create vulkan", "no vulkan devices", "cuda error")
    return any(marker in log.lower() for marker in markers) or returncode in (
        -6, -11, -1073741819, 3221225477, -1073740791, 3221226505,
    )

# 第一步先处理文本型PDF
def extract_text_from_pdf(pdf_path: Path | str) -> list[dict]:
    """使用 pypdf 提取 PDF 已有文字层，保留原始页号。

    Args:
        pdf_path: Path | str：本地 PDF 文件路径。

    Returns:
        list[dict]：每页一个记录，含 source 路径、从 1 开始的 page 和 text；空白页 text 为空字符串。

    Raises:
        ValueError：检测到加密 PDF；读取或 PDF 格式错误向上传递。

    Notes:
        不识别扫描图片；项目内路径保存为相对 PROJECT_ROOT 的路径，项目外保存绝对路径。
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
    """通过 unstructured 的 OCR-only 策略识别 PDF 并按页合并文字。

    Args:
        pdf_path: Path | str：需要中英文 OCR 的本地 PDF。

    Returns:
        list[dict]：按物理页号排序的 source/page/text；没有有效识别文字的页不返回。

    Raises:
        ValueError：partition_pdf 处理失败，并保留原异常链。

    Notes:
        识别语言为 eng 与 chi_sim；需安装对应 OCR 依赖。
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
    """将逐页提取结果组合为带中文页码标题的 Markdown。

    Args:
        pages: list[dict]：各记录需含 page（物理页号）和 text（字符串）。

    Returns:
        str：非空页组成的 Markdown；存在内容时末尾含换行，全部为空时返回空字符串。
    """
    sections = []
    for page in pages:
        text = page["text"].strip()
        if text:
            sections.append(f'## 第 {page["page"]} 页\n\n{text}')
    return "\n\n".join(sections) + ("\n" if sections else "")


def pdf_to_markdown(pdf_path: Path | str, use_ocr: bool = False) -> Path:
    """选择文字层或 OCR 提取，将结果保存为 Markdown 文件。

    Args:
        pdf_path: Path | str：原始 PDF 文件路径。
        use_ocr: bool：True 使用 OCR-only；False 使用 pypdf 文字层，默认 False。

    Returns:
        Path：写入的 papers/extracted/<原文件名>.md 路径。

    Notes:
        自动创建 OUTPUT_DIR；同名输出会被覆盖，原始 PDF 不变。
    """
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
    """在独立进程中运行 MinerU，保存 Markdown、图片及结构化 JSON。

    Args:
        pdf_path: Path | str：存在的 PDF 文件，支持用户目录展开。
        output_dir: Path | str | None：结果目录；None 使用 papers/extracted/mineru，同名结果文件会覆盖。
        tier: str：flash、basic、standard、advanced 之一，默认 standard；首次使用可能下载模型。
        ocr_mode: str：auto 自动选择、txt 使用文字层、ocr 执行识别，默认 auto。
        page_range: str：页面范围，例如 1-3；空字符串处理全部页面，页码从 1 开始。
        device: str：auto 优先 GPU 且设备/内存失败时回退 CPU；gpu 强制 GPU；cpu 强制 CPU。

    Returns:
        Path：输出目录中的 markdown.md 绝对路径。

    Raises:
        FileNotFoundError：PDF 不存在；ValueError：扩展名或设备参数非法；RuntimeError：强制 GPU 不可用、解析失败或未生成 Markdown。

    Notes:
        ONNX 小模型始终使用 CPU。每次尝试写 mineru-gpu.log 或 mineru-cpu.log；下载/输入错误不会被误判成 GPU 错误。

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
    """处理 PDF 转 Markdown 命令行入口与隔离工作进程入口。

    Args:
        无显式输入参数。

    Returns:
        None：主模式打印输出路径；worker 模式将 MinerU 结果写入指定目录。

    Notes:
        无显式参数；从 sys.argv 读取 PDF、输出目录、tier、OCR、页范围和设备。worker 在导入 native 模型前设置环境，避免父进程缓存设备状态。
    """
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
