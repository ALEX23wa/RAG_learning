"""Download immutable official model snapshots without replacing production models."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time

ROOT = Path(__file__).resolve().parents[1]
# Full commits resolved from the publishers' Hugging Face API, 2026-10-02.
MODELS = {
    'bge-m3': {'repository': 'BAAI/bge-m3',
        'revision': '5617a9f61b028005a4858fdac845db406aefb181',
        'directory': 'candidate-bge-m3', 'kind': 'embedding', 'dimension': 1024},
    'qwen-embedding': {'repository': 'Qwen/Qwen3-Embedding-0.6B',
        'revision': '97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3',
        'directory': 'candidate-qwen3-embedding-0.6b', 'kind': 'embedding', 'dimension': 1024},
    'qwen-reranker': {'repository': 'Qwen/Qwen3-Reranker-0.6B',
        'revision': 'e61197ed45024b0ed8a2d74b80b4d909f1255473',
        'directory': 'candidate-qwen3-reranker-0.6b', 'kind': 'reranker'},
}


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def target_path(root, model):
    base = (Path(root) / 'models').resolve()
    name = model['directory']
    target = (base / name).resolve()
    if not name.startswith('candidate-') or target.parent != base:
        raise ValueError('Candidate snapshots must use a direct candidate-* directory')
    if not re.fullmatch(r'[0-9a-f]{40}', model['revision']):
        raise ValueError('A full immutable commit is required')
    return target


def verify_manifest(target, expected=None):
    target = Path(target)
    manifest = json.loads((target / 'snapshot-manifest.json').read_text(encoding='utf-8'))
    if expected and any(manifest.get(k) != expected[k] for k in ('repository', 'revision', 'directory')):
        raise ValueError('Model snapshot identity mismatch')
    if not manifest.get('files') or manifest.get('status') != 'complete':
        raise ValueError('Incomplete model snapshot')
    for name, metadata in manifest['files'].items():
        path = (target / name).resolve()
        if not path.is_relative_to(target.resolve()) or not path.is_file():
            raise ValueError('Missing or unsafe snapshot member: ' + name)
        if path.stat().st_size != metadata['bytes'] or sha256(path) != metadata['sha256']:
            raise ValueError('Model snapshot checksum mismatch: ' + name)
    return manifest


def download(name, state, *, workers=2):
    from huggingface_hub import HfApi, snapshot_download
    spec = MODELS[name]
    target = target_path(state, spec)
    manifest_file = target / 'snapshot-manifest.json'
    if manifest_file.exists():
        return verify_manifest(target, spec)
    info = HfApi().model_info(spec['repository'], revision=spec['revision'])
    if info.sha != spec['revision']:
        raise ValueError('Publisher returned a different commit')
    # BGE-M3 officially publishes pytorch_model.bin, so do not exclude *.bin.
    # Sparse/ColBERT heads are outside this dense-only experiment.
    snapshot_download(spec['repository'], revision=spec['revision'], local_dir=target,
        ignore_patterns=['onnx/*', 'openvino/*', 'imgs/*', '*.jpg', '*.webp', '*.png', '*.pt', '*.h5', '*.ot'],
        max_workers=workers)
    files = {p.relative_to(target).as_posix(): {'bytes': p.stat().st_size, 'sha256': sha256(p)}
             for p in sorted(target.rglob('*')) if p.is_file() and '.cache' not in p.relative_to(target).parts}
    if not any(n.endswith(('.safetensors', '.bin')) for n in files):
        raise ValueError('Snapshot has no model weights')
    manifest = {**spec, 'status': 'complete', 'retrieved_at_unix': time.time(),
                'official_url': 'https://huggingface.co/' + spec['repository'] + '/tree/' + spec['revision'],
                'representation': 'dense' if spec['kind'] == 'embedding' else 'yes-no-logits', 'files': files}
    temporary = manifest_file.with_suffix('.tmp')
    temporary.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    temporary.replace(manifest_file)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('models', nargs='*', choices=list(MODELS))
    parser.add_argument('--state', type=Path, default=ROOT / '.rag')
    parser.add_argument('--proxy', help='Optional explicit HTTP(S) proxy for this process only')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    if args.proxy:
        os.environ['HTTP_PROXY'] = os.environ['HTTPS_PROXY'] = args.proxy
    os.environ.setdefault('HF_HOME', str(args.state / 'huggingface'))
    os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
    for name in args.models or list(MODELS):
        print('Verifying' if args.verify_only else 'Downloading', name, flush=True)
        result = (verify_manifest(target_path(args.state, MODELS[name]), MODELS[name]) if args.verify_only
                  else download(name, args.state, workers=args.workers))
        print(json.dumps({k: result[k] for k in ('repository', 'revision', 'status')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
