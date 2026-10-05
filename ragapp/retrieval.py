import gc
import hashlib
import json
import math
from numbers import Real
import os
import subprocess
import threading
import time
import uuid
from pathlib import Path
import httpx
import re
from .pipeline import embedding_fingerprint, embedding_text, fingerprint
from .model_adapters import embedding_inputs
from .textbudget import windows, token_count

_IMPORT_PROOFS={}


def _import_proof(folder, proof_id, contract):
    path=Path(folder)/('import-'+proof_id+'.json');stat=path.stat()
    stamp=(stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)
    key=(str(path.resolve()),stamp)
    proof=_IMPORT_PROOFS.get(key)
    if proof is None:
        proof=json.loads(path.read_text(encoding='utf-8'))
        if proof.get('fingerprint')!=proof_id or fingerprint({k:v for k,v in proof.items() if k!='fingerprint'})!=proof_id:
            raise ValueError('Imported vector proof checksum mismatch')
        current=path.stat()
        if stamp!=(current.st_dev,current.st_ino,current.st_size,current.st_mtime_ns,current.st_ctime_ns):
            raise ValueError('Imported vector proof changed during read')
        if len(_IMPORT_PROOFS)>16:_IMPORT_PROOFS.clear()
        _IMPORT_PROOFS[key]=proof
    if (proof.get('target_contract')!=contract or proof.get('historical_actual_dtype')!='unknown'
            or proof.get('verification')!='sampled_fp32_consistency' or proof.get('passed') is not True
            or type(proof.get('sample_count')) is not int or proof['sample_count']<32):
        raise ValueError('Imported vector consistency proof is incomplete')
    records=proof.get('sample_records')
    expected_precision={'requested_parameter_dtype':'torch.float32','actual_parameter_dtypes':['torch.float32'],
                        'validation':'all_parameters_checked'}
    if (proof.get('sample_precision')!=expected_precision or not isinstance(records,list)
            or len(records)!=proof['sample_count'] or not all(isinstance(r,dict) for r in records)
            or any(not isinstance(r.get('text_sha256'),str) for r in records)
            or len({r['text_sha256'] for r in records})!=len(records)):
        raise ValueError('Imported vector lacks complete distinct FP32 sample records')
    return proof


def validated_embedding_precision(model):
    actual=sorted({str(parameter.dtype) for parameter in model.parameters()})
    if actual!=['torch.float32']:
        raise ValueError('Embedding requires FP32 parameters; actual dtypes: '+str(actual))
    return {'requested_parameter_dtype':'torch.float32','actual_parameter_dtypes':actual,
            'validation':'all_parameters_checked'}


def file_sha256(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(8*1024*1024),b''):value.update(part)
    return value.hexdigest()


def embedding_model_files(settings):
    """Hash the actual local snapshot once per loaded FP32 session, including tokenizer.

    This authenticates local bytes, not their remote publisher. Candidate import
    additionally verifies the pinned download manifest. Auxiliary HF cache and
    the download manifest itself do not participate in model execution.
    """
    folder=settings.state/'models'/settings.embedding_directory
    files={p.relative_to(folder).as_posix():{'bytes':p.stat().st_size,'sha256':file_sha256(p)}
           for p in sorted(folder.rglob('*')) if p.is_file()
           and '.cache' not in p.relative_to(folder).parts and p.name!='snapshot-manifest.json'}
    if not files:raise ValueError('Embedding snapshot is empty')
    manifest_path=folder/'snapshot-manifest.json'
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
        if (manifest.get('status')!='complete' or manifest.get('files')!=files
                or manifest.get('repository')!=settings.embedding_model
                or manifest.get('revision')!=settings.embedding_revision
                or manifest.get('directory')!=settings.embedding_directory):
            raise ValueError('Embedding snapshot does not match its pinned local manifest')
    elif settings.embedding_directory.startswith('candidate-'):
        raise ValueError('Candidate embedding requires its pinned snapshot manifest')
    return files


def document_cache_contract(settings, model_files_sha256):
    if settings.embedding_precision!='float32':raise ValueError('Verified cache requires explicit FP32')
    return {'embedding_fingerprint':embedding_fingerprint(settings),'model_files_sha256':model_files_sha256,
            'dimension':settings.embedding_dimension,'parameter_dtype':'torch.float32',
            'normalize_embeddings':True,'storage_dtype':'float32',
            'document_template':embedding_inputs(settings,['{text}'],False)[0]}


def checked_document_vector(vector, dimension):
    import numpy as np
    vector=np.asarray(vector)
    if vector.shape!=(dimension,) or vector.dtype!=np.dtype('float32'):
        raise ValueError('Document vector shape/dtype mismatch')
    if not np.isfinite(vector).all() or abs(float(np.linalg.norm(vector))-1.)>2e-4:
        raise ValueError('Document vector must be finite and normalized')
    return vector


def atomic_cache_json(path, value):
    """Bounded replacement with a unique recoverable checkpoint on failure."""
    import tempfile
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent,prefix='.'+path.name+'-',suffix='.tmp',delete=False) as stream:
        temporary=Path(stream.name)
        stream.write(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2).encode('utf-8'))
        stream.flush();os.fsync(stream.fileno())
    for attempt in range(5):
        try:temporary.replace(path);return
        except PermissionError:
            if attempt==4:raise RuntimeError('Cache checkpoint replacement blocked; recover '+str(temporary))
            time.sleep(.05*2**attempt)


def read_document_vector(folder, text, full_input, contract):
    """Only explicit-FP32 caches use sidecars. Never retrofit legacy NPY files."""
    import numpy as np
    key=hashlib.sha256(text.encode('utf-8')).hexdigest()
    binary=Path(folder)/(key+'.npy');marker=binary.with_suffix('.json')
    if not binary.exists() and not marker.exists():return None
    if not binary.is_file() or not marker.is_file():
        raise ValueError('Incomplete verified document cache entry: '+key)
    data=json.loads(marker.read_text(encoding='utf-8'))
    if (data.get('schema')!='verified-document-vector-v1'
            or data.get('fingerprint')!=fingerprint({k:v for k,v in data.items() if k!='fingerprint'})
            or data.get('contract')!=contract or data.get('text_sha256')!=key
            or data.get('input_sha256')!=hashlib.sha256(full_input.encode('utf-8')).hexdigest()
            or data.get('file_sha256')!=file_sha256(binary)):
        raise ValueError('Verified document cache identity/checksum mismatch')
    vector=checked_document_vector(np.load(binary,allow_pickle=False),contract['dimension'])
    if data.get('vector_sha256')!=hashlib.sha256(vector.tobytes()).hexdigest():
        raise ValueError('Document vector checksum mismatch')
    origin=data.get('origin',{})
    if origin.get('kind')=='encoded':
        if origin.get('actual_parameter_dtypes')!=['torch.float32']:
            raise ValueError('Encoded vector lacks actual FP32 precision')
    elif origin.get('kind')=='benchmark_import':
        proof_id=origin.get('proof_fingerprint','')
        if not isinstance(proof_id,str) or not re.fullmatch('[0-9a-f]{64}',proof_id):
            raise ValueError('Imported vector lacks verification proof')
        proof=_import_proof(folder,proof_id,contract)
        if (proof.get('eligible',{}).get(key)!={k:origin.get(k) for k in ('cache_key','vector_sha256')}
                or origin.get('vector_sha256')!=data['vector_sha256']):
            raise ValueError('Imported vector provenance/consistency proof mismatch')
    else:raise ValueError('Unknown verified document vector origin')
    return vector


def write_document_vector(folder, text, full_input, vector, contract, origin):
    """Immutable per-text commit. Retrying identical writes never overwrites conflicts."""
    import numpy as np
    import tempfile
    from filelock import FileLock
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    vector=checked_document_vector(vector,contract['dimension'])
    key=hashlib.sha256(text.encode('utf-8')).hexdigest();binary=folder/(key+'.npy')
    with FileLock(str(folder/'.write.lock'),timeout=30):
        if binary.with_suffix('.json').exists():
            previous=read_document_vector(folder,text,full_input,contract)
            if not np.array_equal(previous,vector):raise ValueError('Conflicting document vector; refusing overwrite')
            return False
        # A crash after binary commit can be completed only with the exact bytes.
        if binary.exists():
            previous=checked_document_vector(np.load(binary,allow_pickle=False),contract['dimension'])
            if not np.array_equal(previous,vector):raise ValueError('Conflicting incomplete document vector')
        else:
            with tempfile.NamedTemporaryFile(dir=folder,prefix='.vector-',suffix='.tmp',delete=False) as stream:
                temporary=Path(stream.name);np.save(stream,vector,allow_pickle=False);stream.flush();os.fsync(stream.fileno())
            temporary.replace(binary)
        marker={'schema':'verified-document-vector-v1','contract':contract,'text_sha256':key,
                'input_sha256':hashlib.sha256(full_input.encode('utf-8')).hexdigest(),
                'file_sha256':file_sha256(binary),'vector_sha256':hashlib.sha256(vector.tobytes()).hexdigest(),
                'origin':origin}
        marker['fingerprint']=fingerprint(marker);atomic_cache_json(binary.with_suffix('.json'),marker)
        return True


def validated_reranker_precision(adapter):
    """Inspect computation parameters, not a score array cast after inference."""
    model = adapter if callable(getattr(adapter, 'parameters', None)) else getattr(adapter, 'model', None)
    if model is None or not callable(getattr(model, 'parameters', None)):
        raise ValueError('Reranker parameter precision is unavailable')
    actual = sorted({str(parameter.dtype) for parameter in model.parameters()})
    if actual != ['torch.float32']:
        raise ValueError('Reranker requires FP32 parameters; actual dtypes: '+str(actual))
    return {'requested_parameter_dtype': 'torch.float32', 'actual_parameter_dtypes': actual,
            'validation': 'all_parameters_checked'}


def validated_rerank_scores(values, expected):
    """Require one finite scalar per input; malformed output never drops candidates."""
    if isinstance(values, (str, bytes, dict)):
        raise ValueError('Reranker must return one finite scalar score per input')
    try:
        scores = list(values)
    except TypeError as exc:
        raise ValueError('Reranker must return one finite scalar score per input') from exc
    if len(scores) != expected:
        raise ValueError(f'Reranker score count mismatch: expected {expected}, got {len(scores)}')
    checked = []
    for score in scores:
        if isinstance(score, bool) or not isinstance(score, Real):
            raise ValueError('Reranker scores must be finite real scalars')
        try:
            value = float(score)
        except (ValueError, OverflowError) as exc:
            raise ValueError('Reranker scores must be finite real scalars') from exc
        if not math.isfinite(value):
            raise ValueError('Reranker scores must be finite real scalars')
        checked.append(value)
    return checked


class Retrieval:
    def __init__(self, settings, store):
        """关联配置与台账，准备检索锁和延迟加载的模型状态。

        Args:
            settings: Settings：目录、索引集合、模型与分块配置。
            store: Store：保存文档版本、文本块、任务和调用台账的 SQLite 存储。

        Returns:
            None：初始化实例；此时不加载模型或连接数据库。
        """
        self.settings=settings; self.store=store; self.lock=threading.RLock()
        self._embedding=None; self._tokenizer=None; self._reranker=None; self.process=None
        self.rerank_lock=threading.RLock()
        self.embedding_precision_record=None;self._document_cache_contract=None

    def start_database(self):
        """连接本机 Qdrant；必要时启动原生进程并创建向量集合。

        Args:
            无显式输入参数。

        Returns:
            None：设置 self.client；新启动进程时同时设置 self.process。

        Raises:
            RuntimeError：缺少 Qdrant 程序或启动后健康检查超时。

        Notes:
            新集合使用 768 维余弦距离，向量、HNSW 和 payload 存储于磁盘。首次启动需要 bootstrap_rag.py 下载的可执行文件。
        """
        from qdrant_client import QdrantClient, models
        try:
            httpx.get(self.settings.qdrant_url+'/healthz',timeout=2).raise_for_status()
        except (httpx.HTTPError, OSError):
            folder=self.settings.state/'qdrant'
            executable=next(folder.rglob('qdrant.exe'),None)
            if executable is None: raise RuntimeError('Run python scripts/bootstrap_rag.py first')
            config=folder/'rag-config.yaml'
            config.write_text('storage:\n  storage_path: '+json.dumps(str(folder/'storage'))+
                              '\n  on_disk_payload: true\nservice:\n  host: 127.0.0.1\n  http_port: 6333\n  grpc_port: 6334\ntelemetry_disabled: true\n',encoding='utf-8')
            with (self.settings.state/'logs'/'qdrant.log').open('ab') as log:
                self.process=subprocess.Popen([str(executable),'--config-path',str(config)],cwd=folder,
                    stdout=log,stderr=log,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            for _ in range(60):
                try:
                    httpx.get(self.settings.qdrant_url+'/healthz',timeout=1).raise_for_status(); break
                except httpx.HTTPError: time.sleep(.5)
            else: raise RuntimeError('Qdrant startup failed; inspect .rag/logs/qdrant.log')
        self.client=QdrantClient(url=self.settings.qdrant_url,timeout=90)
        if not self.client.collection_exists(self.settings.collection):
            self.client.create_collection(self.settings.collection,
                vectors_config=models.VectorParams(size=self.settings.embedding_dimension,distance=models.Distance.COSINE,on_disk=True),
                hnsw_config=models.HnswConfigDiff(on_disk=True),on_disk_payload=True)

    @property
    def tokenizer(self):
        """BM25-only diagnostics need token budgets without loading model weights."""
        if self._embedding is not None:return self._embedding.tokenizer
        with self.lock:
            if self._tokenizer is None:
                from transformers import AutoTokenizer
                self._tokenizer=AutoTokenizer.from_pretrained(
                    str(self.settings.state/'models'/self.settings.embedding_directory),local_files_only=True)
            return self._tokenizer

    @property
    def embedding(self):
        """按需加载并缓存本地 CPU embedding 模型。

        Args:
            无显式输入参数。

        Returns:
            SentenceTransformer：multilingual-e5-base 实例，最大序列长度 512 token。

        Notes:
            仅使用本地模型文件；首次读取该属性会加载权重并将 PyTorch CPU 线程数设为 4。
        """
        if self._embedding is None:
            import torch
            from sentence_transformers import SentenceTransformer
            torch.set_num_threads(4)
            options={}
            if self.settings.embedding_precision=='float32':options['model_kwargs']={'torch_dtype':torch.float32}
            elif self.settings.embedding_precision!='legacy':raise ValueError('Unsupported embedding precision')
            model=SentenceTransformer(str(self.settings.state/'models'/self.settings.embedding_directory),device='cpu',local_files_only=True,**options)
            model.max_seq_length=self.settings.embedding_max_tokens
            if self.settings.embedding_precision=='float32':
                self.embedding_precision_record=validated_embedding_precision(model)
                self._document_cache_contract=document_cache_contract(self.settings,fingerprint(embedding_model_files(self.settings)))
            self._embedding=model
            self._tokenizer=None
        return self._embedding

    def vectors(self, texts, query=False):
        """生成归一化向量，区分检索查询与文档片段。

        Args:
            texts: Sequence[str]：非空文本序列。
            query: bool：True 添加 query: 前缀且不写缓存；False 添加 passage: 前缀并按文本哈希缓存向量。

        Returns:
            numpy.ndarray：形状为 (len(texts), 768) 的向量矩阵，行顺序与输入一致。

        Notes:
            锁内执行以控制模型并发；文档缓存为不允许 pickle 的 .npy 文件。
        """
        with self.lock:
            import numpy as np
            if self.settings.embedding_precision=='float32':return self._fp32_vectors(texts,query)
            if query:
                return self.embedding.encode(embedding_inputs(self.settings,texts,True),normalize_embeddings=True,show_progress_bar=False)
            folder=self.settings.state/'cache'/('vectors-'+embedding_fingerprint(self.settings)); folder.mkdir(parents=True,exist_ok=True)
            paths=[folder/(hashlib.sha256(t.encode()).hexdigest()+'.npy') for t in texts]
            missing=[i for i,p in enumerate(paths) if not p.exists()]
            if missing:
                encoded=self.embedding.encode(embedding_inputs(self.settings,[texts[i] for i in missing],False),normalize_embeddings=True,
                    batch_size=self.settings.embedding_batch,show_progress_bar=False)
                for i,vector in zip(missing,encoded):
                    temporary=paths[i].with_suffix('.tmp')
                    with temporary.open('wb') as stream: np.save(stream,vector)
                    temporary.replace(paths[i])
            return np.stack([np.load(p,allow_pickle=False) for p in paths])

    def _fp32_vectors(self,texts,query):
        import numpy as np
        model=self.embedding
        inputs=embedding_inputs(self.settings,texts,query)
        maximum=min(self.settings.embedding_max_tokens,
                    self.settings.embedding_max_tokens if query else self.settings.embedding_input_tokens)
        if any(len(model.tokenizer(t,add_special_tokens=True,truncation=False)['input_ids'])>maximum for t in inputs):
            raise ValueError('Full embedding input exceeds budget; refusing silent truncation')
        if query:
            encoded=model.encode(inputs,normalize_embeddings=True,show_progress_bar=False)
            if len(encoded)!=len(inputs):raise ValueError('Embedding output row count mismatch')
            return np.stack([checked_document_vector(v,self.settings.embedding_dimension) for v in encoded])
        folder=self.settings.state/'cache'/('vectors-'+embedding_fingerprint(self.settings))
        contract=self._document_cache_contract
        if contract is None:raise ValueError('FP32 embedding model/cache identity was not verified')
        result=[read_document_vector(folder,text,full,contract) for text,full in zip(texts,inputs)]
        positions={}
        for i,vector in enumerate(result):
            if vector is None:positions.setdefault(texts[i],[]).append(i)
        if positions:
            # Encode one representative per cache key. Repeated inference with
            # different padding can differ in FP32; immutable writes stay strict.
            missing=[indices[0] for indices in positions.values()]
            encoded=model.encode([inputs[i] for i in missing],normalize_embeddings=True,
                                 batch_size=self.settings.embedding_batch,show_progress_bar=False)
            if len(encoded)!=len(missing):raise ValueError('Embedding output row count mismatch')
            vectors=[checked_document_vector(v,self.settings.embedding_dimension) for v in encoded]
            for i,vector in zip(missing,vectors):
                write_document_vector(folder,texts[i],inputs[i],vector,contract,
                                      {'kind':'encoded',**self.embedding_precision_record})
                for position in positions[texts[i]]:result[position]=vector
        return np.stack(result)

    def upsert(self, chunks):
        """分批生成文本向量并写入或替换 Qdrant 点。

        Args:
            chunks: Sequence[dict]：文本块列表，每项必须含 id、text、document_id、version。

        Returns:
            None：等待每批写入完成。

        Notes:
            每批最多 48 块；SQLite 当前版本的发布由 Indexer 负责。
        """
        from qdrant_client.models import PointStruct
        for begin in range(0,len(chunks),48):
            batch=chunks[begin:begin+48]
            vectors=self.vectors([embedding_text(c) for c in batch])
            self.client.upsert(self.settings.collection,points=[PointStruct(id=c['id'],vector=v.tolist(),
                payload={'document_id':c['document_id'],'version':c['version']}) for c,v in zip(batch,vectors)],wait=True)

    def remove(self, ids):
        """从当前向量集合删除指定点。

        Args:
            ids: list[str]：需要删除的向量点/文本块 ID。

        Returns:
            None：空列表不发请求；非空列表等待删除完成。
        """
        from qdrant_client.models import PointIdsList
        if ids: self.client.delete(self.settings.collection,PointIdsList(points=ids),wait=True)

    def collect_garbage(self):
        """清理不再对应 SQLite 活跃版本的向量。

        Args:
            无显式输入参数。

        Returns:
            int：本轮删除的陈旧或未发布向量数量。

        Notes:
            每次滚动读取 256 个点；保留 update_failed 文档仍然有效的旧版本。
        """
        offset=None;removed=0
        while True:
            points,offset=self.client.scroll(self.settings.collection,limit=256,offset=offset,
                with_payload=False,with_vectors=False)
            ids=[str(p.id) for p in points];active=self.store.active(ids)
            stale=[i for i in ids if i not in active]
            self.remove(stale);removed+=len(stale)
            if offset is None:break
        return removed

    @property
    def reranker(self):
        with self.rerank_lock:
            if self._reranker is None:
                import torch
                path=str(self.settings.state/'models'/self.settings.reranker_directory)
                if 'Qwen3-Reranker' in self.settings.reranker_model:
                    from .model_adapters import QwenReranker
                    model=QwenReranker(path,self.settings.reranker_max_tokens)
                else:
                    from sentence_transformers import CrossEncoder
                    model=CrossEncoder(path,device='cpu',max_length=self.settings.reranker_max_tokens,
                        local_files_only=True,model_kwargs={'torch_dtype':torch.float32})
                validated_reranker_precision(model)
                self._reranker=model
            return self._reranker

    def _check_generation(self):
        with self.store.db() as db:
            row=db.execute("SELECT value FROM metadata WHERE key='embedding_fingerprint'").fetchone()
            count=db.execute('SELECT count(*) FROM chunks').fetchone()[0]
        if row and row[0]!=embedding_fingerprint(self.settings):
            raise RuntimeError('Embedding configuration does not match this generation; rebuild separately')
        if not row and count:
            original={'embedding_model':'intfloat/multilingual-e5-base',
                      'embedding_revision':'d128750597153bb5987e10b1c3493a34e5a4502a',
                      'embedding_directory':'embedding','embedding_dimension':768,'embedding_max_tokens':512,
                      'embedding_precision':'legacy'}
            if any(getattr(self.settings,k)!=v for k,v in original.items()):
                raise RuntimeError('Legacy generation requires its original pinned E5 embedding configuration')

    def rank(self, query, rows, trace=None):
        """Window both sides without silent truncation; max relevance per candidate."""
        if not rows:return []
        with self.rerank_lock:
            model=self.reranker
            tokenizer=getattr(model,'tokenizer',None)
            pairs=[];owners=[]
            maximum=self.settings.reranker_max_tokens
            qwen='Qwen3-Reranker' in self.settings.reranker_model
            adapter_max=getattr(model,'maximum' if qwen else 'max_length',None)
            if type(adapter_max) is int and adapter_max>0:maximum=min(maximum,adapter_max)
            if trace is not None:
                trace.update(rerank_max_pair_tokens=None,rerank_model_max_tokens=maximum,
                    rerank_pair_budget_verified=False,silent_truncations=None)
            for number,row in enumerate(rows):
                document=Path(row['path']).name+'\n'+embedding_text(row)
                if tokenizer is None: # deterministic injected test adapters
                    pairs.append((query,document));owners.append(number);continue
                overhead=128 if qwen else 8
                for q,_,_ in windows(tokenizer,query,min(self.settings.query_tokens,maximum//3)):
                    budget=maximum-token_count(tokenizer,q)-overhead
                    for d,_,_ in windows(tokenizer,document,budget, min(32,budget//4)):
                        pairs.append((q,d));owners.append(number)
            if tokenizer is not None:
                maximum_observed=0
                for q,d in pairs:
                    if qwen:
                        from .model_adapters import TASK
                        text=model.prefix+f'<Instruct>: {TASK}\n<Query>: {q}\n<Document>: {d}'+model.suffix
                        encoded=tokenizer(text,add_special_tokens=True,truncation=False)
                    else:
                        encoded=tokenizer(q,d,add_special_tokens=True,truncation=False)
                    length=len(encoded['input_ids'])
                    maximum_observed=max(maximum_observed,length)
                    if trace is not None:trace['rerank_max_pair_tokens']=maximum_observed
                    if length>maximum:
                        raise ValueError(f'Reranker pair input exceeds budget before inference: {length} > {maximum}; no truncation allowed')
                if trace is not None:
                    trace['rerank_pair_budget_verified']=True
                    trace['silent_truncations']=0
            predictions=model.predict(pairs,batch_size=4,show_progress_bar=False)
        predictions=validated_rerank_scores(predictions,len(pairs))
        scores=[None]*len(rows)
        for owner,score in zip(owners,predictions):
            scores[owner]=score if scores[owner] is None else max(scores[owner],score)
        if any(score is None for score in scores):
            raise ValueError('Reranker has no input window for a candidate')
        if trace is not None:
            trace['rerank_pairs']=len(pairs);trace['rerank_windowed']=len(pairs)>len(rows)
        return scores

    def search(self, query, limit=None, *, document_ids=None, trace=None, options=None):
        """Explicit scopes, independent recall lanes, RRF, then token-budgeted reranking."""
        self._check_generation()
        t=time.perf_counter();trace=trace if trace is not None else {};options=options or {}
        limit=limit or self.settings.evidence_limit
        def setting(name):return options.get(name,getattr(self.settings,name))
        focused,hint_ids=self.store.focused(query)
        source_ids=self.store.source_scope(query,document_ids)
        trace.update({'query':query,'explicit_document_ids':sorted(source_ids) if source_ids is not None else None,
                      'hint_document_ids':sorted(hint_ids or []),'stages':{}})
        if source_ids is not None and not source_ids:return []
        labels={'审稿编辑':'Review Editors','协调主要作者':'Coordinating Lead Authors',
                '英文标题':'title','引脚':'pin','工作温度':'operating temperature',
                '储存温度':'storage temperature','封装':'package','感光单元':'image sensing elements'}
        parts=[part for part,_,_ in windows(self.tokenizer,query,self.settings.query_tokens)]
        if not parts:return []
        trace['query_windows']=len(parts)
        trace['query_input_tokens']=token_count(self.tokenizer,query)
        trace['query_window_tokens']=[token_count(self.tokenizer,p) for p in parts]
        trace['query_window_budget']=self.settings.query_tokens
        trace['embedding_model_max_tokens']=self.settings.embedding_max_tokens
        lanes=[];lane_names=[];dense_all=[];filtered=0
        mode=setting('retrieval_mode')
        start=time.perf_counter()
        for part in parts:
            expanded=part+' '+' '.join(v for k,v in labels.items() if k in part)
            if mode!='dense':
                lanes.append(self.store.lexical(expanded,setting('lexical_candidates'),source_ids));lane_names.append('bm25')
        trace['stages']['bm25']=time.perf_counter()-start
        if mode!='bm25':
            from qdrant_client.models import Filter,FieldCondition,MatchAny
            query_filter=Filter(must=[FieldCondition(key='document_id',match=MatchAny(any=sorted(source_ids)))]) if source_ids else None
            start=time.perf_counter();vectors=self.vectors(parts,query=True)
            trace['stages']['query_embedding']=time.perf_counter()-start;start=time.perf_counter()
            for vector in vectors:
                desired=setting('dense_candidates');take=min(desired*2,self.settings.dense_fetch_limit)
                while True:
                    hits=self.client.query_points(self.settings.collection,query=vector.tolist(),limit=take,query_filter=query_filter).points
                    ids=[str(p.id) for p in hits];active=self.store.active(ids)
                    dense=[i for i in ids if i in active and (source_ids is None or active[i]['document_id'] in source_ids)]
                    if len(dense)>=desired or len(hits)<take or take>=self.settings.dense_fetch_limit:break
                    take=min(take*2,self.settings.dense_fetch_limit)
                filtered+=len(ids)-len(dense);dense_all.extend(dense)
                lanes.append(dense[:desired]);lane_names.append('dense')
            trace['stages']['dense']=time.perf_counter()-start
        if mode.startswith('hybrid') and hint_ids:
            # Hints remain a soft lane; they can never eliminate global candidates.
            soft=focused or self.store.lexical(query,20,hint_ids)
            lanes.append(soft);lane_names.append('focused')
            # Sparse wording can miss a definition within the named source.
            # Reuse the original query vector for a bounded, source-biased dense
            # lane; global BM25/dense lanes and their candidates remain present.
            hint_scope=hint_ids if source_ids is None else hint_ids & source_ids
            if hint_scope:
                start=time.perf_counter()
                hint_filter=Filter(must=[FieldCondition(key='document_id',match=MatchAny(any=sorted(hint_scope)))])
                for vector in vectors:
                    hits=self.client.query_points(self.settings.collection,query=vector.tolist(),
                        limit=min(40,self.settings.dense_fetch_limit),query_filter=hint_filter).points
                    ids=[str(p.id) for p in hits];valid=self.store.active(ids)
                    lanes.append([i for i in ids if i in valid and valid[i]['document_id'] in hint_scope][:20])
                    lane_names.append('hint_dense')
                trace['stages']['hint_dense']=time.perf_counter()-start
        active=self.store.active(list(dict.fromkeys(i for lane in lanes for i in lane)))
        if source_ids is not None:active={i:r for i,r in active.items() if r['document_id'] in source_ids}
        exact=bool(focused and source_ids and re.search(r'第\s*\d+\s*行|\b\d{4}-\d{2}-\d{2}\b',query))
        if exact:active={i:r for i,r in active.items() if i in focused}
        scores={};ranks={}
        for lane_name,lane in zip(lane_names,lanes):
            for rank,key in enumerate(dict.fromkeys(lane),1):
                if key in active:
                    scores[key]=scores.get(key,0)+1/(setting('rrf_k')+rank)
                    ranks.setdefault(key,{})[lane_name]=min(rank,ranks.get(key,{}).get(lane_name,rank))
        fused=sorted(scores,key=lambda i:(-scores[i],i))
        candidates=fused[:setting('rerank_candidates')]
        trace['filtered_dense']=filtered;trace['candidate_count']=len(fused)
        trace['candidates']=[{**active[i],'locator':json.loads(active[i]['locator']),
                              'rrf_score':scores[i],'ranks':ranks[i]} for i in fused[:max(80,setting('rerank_candidates'))]]
        if not candidates:return []
        start=time.perf_counter()
        ranked=self.rank(query,[active[i] for i in candidates],trace) if mode=='hybrid_rerank' else [scores[i] for i in candidates]
        trace['stages']['rerank']=time.perf_counter()-start
        chosen=sorted(zip(candidates,ranked),key=lambda pair:float(pair[1]),reverse=True)[:limit]
        still_active=self.store.active([i for i,_ in chosen]);results=[]
        for i,score in chosen:
            if i not in still_active:continue
            row=still_active[i];locator=json.loads(row['locator'])
            if 'line_start' in locator and Path(row['path']).suffix.lower() in {'.py','.java','.cpp','.c','.h'}:
                matches={}
                for symbol in re.findall(r'\b[A-Z][A-Z_]{3,}\b',query):
                    match=re.search(r'(?m)^[ \t]*'+re.escape(symbol)+r'\s*(?::[^=\n]+)?=(?!=)',row['text'])
                    if match:matches[symbol]=locator['line_start']+row['text'][:match.start()].count('\n')
                if matches:locator['matched_lines']=matches
            results.append({**row,'locator':locator,'score':float(score),'retrieval':{'rrf_score':scores[i],'ranks':ranks[i]}})
        trace['elapsed']=time.perf_counter()-t
        return results

