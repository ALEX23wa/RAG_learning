"""Record/verify present-day baseline assets without loading models or writing Qdrant.

This supplemental record explicitly does not claim assets were hashed when the
original code snapshot was taken. Qdrant is read in small ID-ordered batches.
"""
import argparse
import base64
import contextlib
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import struct

import httpx

ROOT = Path(__file__).resolve().parents[1]
COLLECTION = 'papers_e5_v1'


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def fingerprint(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _update(digest, value):
    raw = canonical(value)
    digest.update(struct.pack('>Q', len(raw))); digest.update(raw)


def catalog_signature(path, reference=None):
    """Hash schema and ordered logical rows, including FTS content/index tables.

    Newly added empty tables are allowed in a working copy initialized by Store;
    every table in the reference keeps exactly the same schema and content.
    """
    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError('Baseline catalog is missing: ' + str(path))
    tables = {}
    with contextlib.closing(sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)) as db:
        db.execute('BEGIN')
        schemas = dict(db.execute("SELECT name,sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"))
        expected = set(reference['tables']) if reference else set(schemas)
        if not expected <= set(schemas):
            raise ValueError('Baseline catalog reference tables are missing')
        for name in sorted(set(schemas) - expected):
            quoted = '"' + name.replace('"', '""') + '"'
            if db.execute('SELECT 1 FROM ' + quoted + ' LIMIT 1').fetchone():
                raise ValueError('Unexpected nonempty table in baseline catalog copy: ' + name)
        for name in sorted(expected):
            quoted = '"' + name.replace('"', '""') + '"'
            columns = [list(row) for row in db.execute('PRAGMA table_info(' + quoted + ')')]
            selected = ','.join('"' + row[1].replace('"', '""') + '"' for row in columns)
            try:
                cursor = db.execute('SELECT rowid,' + selected + ' FROM ' + quoted + ' ORDER BY rowid')
                includes_rowid = True
            except sqlite3.OperationalError as exc:
                if 'rowid' not in str(exc).lower():
                    raise
                cursor = db.execute('SELECT ' + selected + ' FROM ' + quoted + ' ORDER BY ' + selected)
                includes_rowid = False
            digest, count = hashlib.sha256(), 0
            for row in cursor:
                values = [{'sqlite_blob_base64': base64.b64encode(value).decode('ascii')} if isinstance(value, bytes)
                          else value for value in row]
                _update(digest, values); count += 1
            tables[name] = {'sql': schemas[name], 'columns': columns, 'includes_rowid': includes_rowid,
                            'rows': count, 'sha256': digest.hexdigest()}
    result = {'tables': tables, 'sha256': fingerprint(tables)}
    if reference and result != reference:
        raise ValueError('Baseline catalog logical contents changed')
    return result


def model_signature(state):
    models = {}
    for name in ('embedding', 'reranker'):
        directory = Path(state) / 'models' / name
        if not directory.is_dir():
            raise ValueError('Baseline model directory is missing: ' + name)
        files = {}
        for path in sorted(directory.rglob('*')):
            relative = path.relative_to(directory)
            if '.cache' in relative.parts or '.git' in relative.parts or not path.is_file():
                continue
            if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()):
                raise ValueError('Unsafe baseline model member: ' + str(relative))
            before = path.stat()
            value = file_hash(path)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                raise ValueError('Baseline model changed during hashing: ' + str(relative))
            files[relative.as_posix()] = {'bytes': after.st_size, 'sha256': value}
        if not files or not any(key.endswith(('.safetensors', '.bin')) for key in files):
            raise ValueError('Baseline model weights are missing: ' + name)
        models[name] = {'files': files, 'sha256': fingerprint(files)}
    return models


def qdrant_signature(url, collection=COLLECTION, batch_size=16, client=None):
    """Hash actual vectors and payloads with bounded memory, sorted by point ID."""
    if not 1 <= batch_size <= 128:
        raise ValueError('Qdrant hash batch size must be between 1 and 128')
    def read(connection):
        endpoint = url.rstrip('/') + '/collections/' + collection
        response = connection.get(endpoint); response.raise_for_status()
        info = response.json()['result']
        configuration = info['config']
        response = connection.post(endpoint + '/points/count', json={'exact': True}); response.raise_for_status()
        expected_count = response.json()['result']['count']
        digest, count, offset, previous = hashlib.sha256(), 0, None, None
        seen_offsets = set()
        while True:
            body = {'limit': batch_size, 'with_payload': True, 'with_vector': True}
            if offset is not None:
                body['offset'] = offset
            response = connection.post(endpoint + '/points/scroll', json=body); response.raise_for_status()
            page = response.json()['result']
            for point in page['points']:
                identifier = point['id']
                key = (0, identifier) if type(identifier) is int else (1, str(identifier).lower())
                if previous is not None and key <= previous:
                    raise ValueError('Qdrant scroll did not preserve unique ascending point IDs')
                if 'vector' not in point or point['vector'] is None:
                    raise ValueError('Qdrant scroll omitted actual vector data')
                _update(digest, {'id': identifier, 'vector': point['vector'], 'payload': point.get('payload')})
                previous, count = key, count + 1
            offset = page.get('next_page_offset')
            if offset is None:
                break
            encoded = canonical(offset)
            if encoded in seen_offsets:
                raise ValueError('Qdrant scroll repeated an offset')
            seen_offsets.add(encoded)
        response = connection.post(endpoint + '/points/count', json={'exact': True}); response.raise_for_status()
        if count != expected_count or response.json()['result']['count'] != expected_count:
            raise ValueError('Qdrant collection changed while its assets were being read')
        response = connection.get(endpoint); response.raise_for_status()
        if response.json()['result']['config'] != configuration:
            raise ValueError('Qdrant configuration changed during hashing')
        return {'collection': collection, 'points': count, 'points_sha256': digest.hexdigest(),
                'configuration': configuration, 'configuration_sha256': fingerprint(configuration),
                'algorithm': 'sha256 of length-prefixed canonical id/vector/payload JSON in ascending ID order'}
    if client is not None:
        return read(client)
    with httpx.Client(timeout=120) as connection:
        return read(connection)


def load_manifest(snapshot):
    path = Path(snapshot) / 'asset-manifest.json'
    if not path.is_file():
        raise ValueError('Baseline assets not recorded; run scripts/freeze_baseline_assets.py freeze first')
    value = json.loads(path.read_text(encoding='utf-8'))
    if value.get('schema') != 'baseline-assets-v1' or value.get('original_snapshot_had_asset_hashes') is not False:
        raise ValueError('Unsupported or misleading baseline asset manifest')
    return value


def verify_assets(snapshot, state, url='http://127.0.0.1:6333', *, working_catalog=None, batch_size=16):
    snapshot, state = Path(snapshot), Path(state)
    manifest = load_manifest(snapshot)
    if file_hash(snapshot / 'snapshot-manifest.json') != manifest['original_code_manifest_sha256']:
        raise ValueError('Original baseline code manifest changed')
    catalog_signature(snapshot / 'catalog.sqlite3', manifest['catalog'])
    if working_catalog is not None:
        catalog_signature(working_catalog, manifest['catalog'])
    if model_signature(state) != manifest['models']:
        raise ValueError('Baseline model files changed')
    if qdrant_signature(url, manifest['qdrant']['collection'], batch_size) != manifest['qdrant']:
        raise ValueError('Baseline Qdrant vectors, payloads or configuration changed')
    return manifest


def freeze(snapshot, state, url='http://127.0.0.1:6333', *, batch_size=16):
    snapshot, state = Path(snapshot), Path(state)
    target = snapshot / 'asset-manifest.json'
    if target.exists():
        return verify_assets(snapshot, state, url, batch_size=batch_size)
    code_manifest = json.loads((snapshot / 'snapshot-manifest.json').read_text(encoding='utf-8'))
    for entry in code_manifest['files']:
        if file_hash(snapshot / entry['path']) != entry['sha256']:
            raise ValueError('Original frozen code/file changed: ' + entry['path'])
    started = datetime.now(timezone.utc).isoformat()
    catalog = catalog_signature(snapshot / 'catalog.sqlite3')
    models = model_signature(state)
    qdrant = qdrant_signature(url, batch_size=batch_size)
    # A second independent pass detects mutation across the read window.
    catalog_signature(snapshot / 'catalog.sqlite3', catalog)
    if model_signature(state) != models or qdrant_signature(url, batch_size=batch_size) != qdrant:
        raise ValueError('Baseline assets changed during their initial recording')
    value = {'schema': 'baseline-assets-v1', 'recorded_started_utc': started,
        'recorded_completed_utc': datetime.now(timezone.utc).isoformat(),
        'original_snapshot_had_asset_hashes': False,
        'provenance_note': 'These asset hashes were recorded after the original code snapshot; they establish the present baseline only, not historical asset identity.',
        'original_code_manifest_sha256': file_hash(snapshot / 'snapshot-manifest.json'),
        'catalog': catalog, 'models': models, 'qdrant': qdrant,
        'model_inventory_excludes': ['.cache', '.git'],
        'verification_note': 'Qdrant was read twice without writes or snapshots; subsequent runs verify before and after. No model was loaded.'}
    # Exclusive creation prevents silently overwriting an existing baseline.
    with target.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('freeze', 'verify'))
    parser.add_argument('--snapshot', type=Path, default=ROOT / '.rag/experiments/quality-v2-baseline')
    parser.add_argument('--state', type=Path, default=ROOT / '.rag')
    parser.add_argument('--url', default='http://127.0.0.1:6333')
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    operation = freeze if args.command == 'freeze' else verify_assets
    value = operation(args.snapshot, args.state, args.url, batch_size=args.batch_size)
    print(json.dumps({'verified': True, 'asset_manifest_sha256': file_hash(args.snapshot / 'asset-manifest.json'),
        'catalog_sha256': value['catalog']['sha256'], 'points': value['qdrant']['points'],
        'points_sha256': value['qdrant']['points_sha256'], 'recorded_completed_utc': value['recorded_completed_utc'],
        'original_snapshot_had_asset_hashes': False}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
