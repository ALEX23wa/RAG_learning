"""Asset provenance checks use tiny local fixtures, never models or real Qdrant."""
from contextlib import closing
import sqlite3

import pytest

from scripts.freeze_baseline_assets import catalog_signature, model_signature, qdrant_signature


def test_catalog_freeze_covers_fts_and_allows_only_empty_extra_schema(tmp_path):
    original, copy = tmp_path / 'original.sqlite3', tmp_path / 'copy.sqlite3'
    with closing(sqlite3.connect(original)) as db:
        db.execute('CREATE TABLE chunks(id TEXT PRIMARY KEY,text TEXT)')
        db.execute('INSERT INTO chunks VALUES(?,?)', ('a', 'original evidence'))
        db.execute('CREATE VIRTUAL TABLE chunks_fts USING fts5(id UNINDEXED,tokens)')
        db.execute('INSERT INTO chunks_fts VALUES(?,?)', ('a', 'original evidence'))
        db.commit()
    signature = catalog_signature(original)
    with closing(sqlite3.connect(original)) as source, closing(sqlite3.connect(copy)) as destination:
        source.backup(destination)
    with closing(sqlite3.connect(copy)) as db:
        db.execute('CREATE TABLE extra_empty(id TEXT)'); db.commit()
    assert catalog_signature(copy, signature) == signature
    with closing(sqlite3.connect(copy)) as db:
        db.execute("INSERT INTO extra_empty VALUES('unexpected')"); db.commit()
    with pytest.raises(ValueError, match='Unexpected nonempty'):
        catalog_signature(copy, signature)
    with closing(sqlite3.connect(copy)) as db:
        db.execute('DELETE FROM extra_empty')
        db.execute("UPDATE chunks_fts SET tokens='tampered' WHERE id='a'"); db.commit()
    with pytest.raises(ValueError, match='logical contents changed'):
        catalog_signature(copy, signature)


def test_actual_model_bytes_are_frozen_but_downloader_cache_is_excluded(tmp_path):
    for name in ('embedding', 'reranker'):
        directory = tmp_path / 'models' / name; directory.mkdir(parents=True)
        (directory / 'model.safetensors').write_bytes(b'weight bytes')
        (directory / 'config.json').write_text('{}')
    models = model_signature(tmp_path)
    cache = tmp_path / 'models/embedding/.cache'; cache.mkdir()
    (cache / 'download.metadata').write_text('runtime metadata')
    assert model_signature(tmp_path) == models
    (tmp_path / 'models/embedding/model.safetensors').write_bytes(b'different weights')
    assert model_signature(tmp_path) != models


class Response:
    def __init__(self, value): self.value = value
    def raise_for_status(self): pass
    def json(self): return {'result': self.value}


class Client:
    def __init__(self, mutate=False, missing_vector=False, duplicate=False):
        self.mutate, self.missing_vector, self.duplicate = mutate, missing_vector, duplicate
    def get(self, _url):
        return Response({'config': {'params': {'vectors': {'size': 2, 'distance': 'Cosine'}}}})
    def post(self, url, json):
        if url.endswith('/count'):
            return Response({'count': 2})
        first = {'id': 1, 'vector': [.1, .2], 'payload': {'x': 'a'}}
        second = {'id': 1 if self.duplicate else 2, 'vector': [.3, .4],
                  'payload': {'x': 'changed' if self.mutate else 'b'}}
        if self.missing_vector:
            first.pop('vector')
        return Response({'points': [second] if json.get('offset') == 2 else [first],
                         'next_page_offset': None if json.get('offset') == 2 else 2})


def test_qdrant_hashes_actual_vectors_and_payloads_in_stable_small_pages():
    first = qdrant_signature('http://test', client=Client())
    assert first == qdrant_signature('http://test', client=Client(), batch_size=1)
    assert first['points'] == 2
    assert first != qdrant_signature('http://test', client=Client(mutate=True))
    with pytest.raises(ValueError, match='omitted actual vector'):
        qdrant_signature('http://test', client=Client(missing_vector=True))
    with pytest.raises(ValueError, match='ascending point IDs'):
        qdrant_signature('http://test', client=Client(duplicate=True))
