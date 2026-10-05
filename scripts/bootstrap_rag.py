"""Download the pinned native database and local model snapshots into .rag."""
from pathlib import Path
import hashlib
import json
import os
import zipfile
import httpx

ROOT = Path(__file__).resolve().parents[1] / '.rag'
os.environ.setdefault('HF_HOME', str(ROOT / 'huggingface'))

def main():
    """下载固定版本的 Windows Qdrant 和两个本地模型快照。

    Args:
        无显式输入参数。

    Returns:
        None：资源写入项目 .rag，进度打印到标准输出。

    Notes:
        无显式参数；使用模块 ROOT。校验 ZIP 成员路径防止越界，记录下载哈希和模型 revision；现有 Qdrant 可执行文件不重复下载。网络错误向上传递。
    """
    from huggingface_hub import snapshot_download
    ROOT.mkdir(exist_ok=True)
    target = ROOT / 'qdrant'
    target.mkdir(exist_ok=True)
    url = 'https://github.com/qdrant/qdrant/releases/download/v1.19.1/qdrant-x86_64-pc-windows-msvc.zip'
    if not (target / 'qdrant.exe').exists():
        archive = target / 'download.zip'
        with httpx.stream('GET', url, follow_redirects=True, timeout=180) as response:
            response.raise_for_status()
            with archive.open('wb') as stream:
                for chunk in response.iter_bytes():
                    stream.write(chunk)
        with zipfile.ZipFile(archive) as package:
            for member in package.infolist():
                if not (target / member.filename).resolve().is_relative_to(target.resolve()):
                    raise ValueError('Unsafe archive member')
            package.extractall(target)
        (target / 'download.json').write_text(json.dumps({'url': url, 'sha256': hashlib.sha256(archive.read_bytes()).hexdigest()}))
    models=[('intfloat/multilingual-e5-base','embedding','d128750597153bb5987e10b1c3493a34e5a4502a'),
            ('BAAI/bge-reranker-v2-m3','reranker','953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e')]
    for repo, name,revision in models:
        print('Downloading', repo, flush=True)
        snapshot_download(repo, revision=revision, local_dir=ROOT / 'models' / name,
                          ignore_patterns=['*.onnx', 'onnx/*', 'openvino/*', '*.bin', '*.h5', '*.ot'], max_workers=3)
    (ROOT/'models'/'manifest.json').write_text(json.dumps([
        {'repository':repo,'directory':name,'revision':revision} for repo,name,revision in models],indent=2),encoding='utf-8')
    print('Resources ready', flush=True)

if __name__ == '__main__':
    main()
