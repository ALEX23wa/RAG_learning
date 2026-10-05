import hashlib
import json
import os
import re
import shutil
import threading
import time
import uuid
from pathlib import Path
from .parsers import SUPPORTED, parse, cached_json, save_json
from .store import terms
from .pipeline import index_fingerprint, parsing_fingerprint, embedding_fingerprint, embedding_text
from .structure import token_offsets, structural_chunks
from .model_adapters import embedding_inputs

def digest(path):
    """流式计算文件内容的 SHA-256，用于增量更新和版本检查。

    Args:
        path: Path：可读取的文件路径。

    Returns:
        str：64 个十六进制字符的文件内容摘要。

    Notes:
        每次读取 1 MiB，避免一次将大型文件装入内存。
    """
    h=hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''): h.update(part)
    return h.hexdigest()

class Indexer:
    def __init__(self,settings,store,retrieval,models=None):
        """关联同步所需组件并初始化单进程同步锁。

        Args:
            settings: Settings：目录、索引集合、模型与分块配置。
            store: Store：保存文档版本、文本块、任务和调用台账的 SQLite 存储。
            retrieval: Retrieval：本地向量化、Qdrant 写入与混合检索组件。
            models: Models | None：可选在线模型组件；提供时可为独立图片生成视觉描述。

        Returns:
            None：仅初始化引用和锁，不扫描资料。
        """
        self.settings=settings; self.store=store; self.retrieval=retrieval
        self.lock=threading.Lock()
        self.models=models

    def _prepare_generation(self):
        """Refuse to mix structure/model generations in a populated catalog."""
        expected=index_fingerprint(self.settings)
        with self.store.db() as db:
            saved=db.execute("SELECT value FROM metadata WHERE key='index_fingerprint'").fetchone()
            populated=db.execute('SELECT 1 FROM chunks LIMIT 1').fetchone()
            if saved and saved[0]!=expected:
                raise RuntimeError('Index fingerprint changed; rebuild into a new generation before syncing')
            if not saved and populated:
                raise RuntimeError('Legacy index has no pipeline fingerprint; rebuild into a new generation before syncing')
            values={'index_fingerprint':expected,'parsing_fingerprint':parsing_fingerprint(self.settings),
                    'embedding_fingerprint':embedding_fingerprint(self.settings),
                    'index_profile':self.settings.index_profile,'index_schema':self.settings.index_schema}
            db.executemany('INSERT OR REPLACE INTO metadata(key,value) VALUES(?,?)',values.items())
        return expected

    def _embedding_tokens(self,text,tokenizer,query=False):
        """Count the actual adapter input, including tokenizer special tokens."""
        full=embedding_inputs(self.settings,[text],query)[0]
        try:
            encoded=tokenizer(full,add_special_tokens=True,truncation=False)
        except TypeError:
            # Minimal test tokenizers may not expose the production keyword API.
            try:encoded=tokenizer(full,add_special_tokens=True)
            except TypeError:encoded=tokenizer(full)
        if 'input_ids' in encoded:return len(encoded['input_ids'])
        # Offset-only fixtures have no special tokens unless explicitly declared.
        special=tokenizer.num_special_tokens_to_add(pair=False) if hasattr(tokenizer,'num_special_tokens_to_add') else 0
        return len([s for s in encoded['offset_mapping'] if s[1]>s[0]])+special

    def _retrieval_text(self,item,tokenizer,query=False):
        """Fit decoration to the complete model-input budget, preserving body."""
        budget=self.settings.embedding_input_tokens
        body=item['text'];text=embedding_text(item)
        body_tokens=self._embedding_tokens(body,tokenizer,query)
        if body_tokens>budget:
            raise ValueError('Source chunk plus embedding template/special tokens exceeds input budget; rechunk first')
        if self._embedding_tokens(text,tokenizer,query)<=budget:return text
        prefix=text[:-len(body)].rstrip() if body and text.endswith(body) else ''
        offsets=token_offsets(tokenizer,prefix)
        keep=min(len(offsets),max(0,budget-body_tokens-2))
        while keep:
            candidate=prefix[:offsets[keep-1][1]]+'\n'+body
            if self._embedding_tokens(candidate,tokenizer,query)<=budget:return candidate
            keep-=1
        return body

    def _chunks(self,blocks,tokenizer):
        """Shrink source windows at raw offsets only when full inputs need it."""
        fits=lambda text:self._embedding_tokens(text,tokenizer)<=self.settings.embedding_input_tokens
        yield from structural_chunks(blocks,tokenizer,self.settings.chunk_tokens,
                                     self.settings.overlap,input_fits=fits)

    def scan(self):
        """递归收集允许索引的原始资料，并按处理成本排序。

        Args:
            无显式输入参数。

        Returns:
            list[Path]：支持格式的文件路径；小文件和文字 PDF 优先。

        Raises:
            RuntimeError：资料根目录不可访问；目录遍历的 OSError 也会向上传递，避免把读取失败误判为删除。

        Notes:
            排除隐藏目录、派生输出、临时文件、密钥命名文件和符号链接。使用实例 corpus，不修改磁盘文件。
        """
        root=self.settings.corpus
        if not root.is_dir(): raise RuntimeError('Corpus root unavailable; deletions were NOT applied')
        result=[]
        def fail(error):
            """让目录遍历错误立即中止扫描，阻止错误删除判断。

            Args:
                error: OSError：os.walk 传给 onerror 回调的目录读取异常。

            Returns:
                None：不正常返回，重新抛出输入异常。
            """
            raise error
        for folder,dirs,files in os.walk(root,onerror=fail,followlinks=False):
            dirs[:]=[d for d in dirs if not d.startswith('.') and d not in {'extracted','__pycache__','node_modules'}
                     and not (Path(folder)/d).is_symlink()]
            for name in files:
                path=Path(folder)/name
                if name.startswith(('.', '~','$')) or path.is_symlink(): continue
                if re.search(r'(?i)(^|[._-])(secrets?|credentials?|api[_-]?key|private[_-]?key)([._-]|$)',name):continue
                if path.suffix.lower() in SUPPORTED and path.resolve().is_relative_to(root.resolve()): result.append(path)
        # Publish small/text documents first; a long scanned book must not delay everything.
        def priority(path):
            """估算文件处理成本，生成同步排序键。

            Args:
                path: Path：已通过扫描规则的原文件。

            Returns:
                tuple[int, int, int, str]：格式/文字质量优先级、页数、文件字节数、路径；数值较小者先处理。

            Notes:
                PDF 抽样首/中/近末页；读取异常时放到末尾，由正式解析记录错误。
            """
            if path.suffix.lower()!='.pdf':return (0,0,path.stat().st_size,str(path))
            try:
                from pypdf import PdfReader
                reader=PdfReader(path)
                samples={0,len(reader.pages)//2,max(0,len(reader.pages)-2)}
                quality=sum(len(reader.pages[i].extract_text() or '') for i in samples)/max(len(samples),1)
                return (1 if quality>400 or len(reader.pages)<=20 else 2,len(reader.pages),path.stat().st_size,str(path))
            except Exception:return (3,0,path.stat().st_size,str(path))
        return sorted(result,key=priority)

    def sync(self,job_id=None):
        """按内容哈希执行增量入库、改名识别、删除屏蔽和版本发布。

        Args:
            job_id: str | None：复用的任务 ID；省略时新建任务。

        Returns:
            dict：job_id、completed/unchanged/failed/deleted/renamed/duplicates 等计数与错误明细；completed 指本轮重新入库数。

        Raises:
            RuntimeError：已有同步任务、根目录不可访问等全局错误；单文件异常记录到返回值而不终止整轮。

        Notes:
            同时使用进程文件锁和线程锁。先写向量再提交台账新版本；单文件失败保留旧版本并继续其他文件，运行状态持续落库。可能创建解析缓存及调用图片描述模型。
        """
        from filelock import FileLock, Timeout
        process_lock=FileLock(str(self.settings.state/'sync.lock'),timeout=0)
        try: process_lock.acquire()
        except Timeout: raise RuntimeError('Sync already running in another process')
        if not self.lock.acquire(blocking=False):
            process_lock.release()
            raise RuntimeError('Sync already running')
        try:
            generation_fingerprint=self._prepare_generation()
        except Exception:
            self.lock.release();process_lock.release();raise
        job_id=self.store.job('running',job_id=job_id)
        progress={'completed':0,'unchanged':0,'failed':0,'deleted':0,'renamed':0,'duplicates':0,'errors':[]}
        try:
            files=self.scan(); progress['total']=len(files)
            bypath={d['path']:d for d in self.store.documents()}
            renamed_ids=set()
            seen={str(p.relative_to(self.settings.corpus)) for p in files}
            # Hide deleted files before expensive parsing begins.
            with self.store.db() as db:
                for name,d in bypath.items():
                    if name not in seen and d['status']!='deleted' and not d['excluded']:
                        db.execute("UPDATE documents SET status='deleted',updated=? WHERE id=?",(time.time(),d['id']))
                        progress['deleted']+=1
            tokenizer=self.retrieval.embedding.tokenizer
            for path in files:
                name=str(path.relative_to(self.settings.corpus)); old=bypath.get(name)
                if old and old['excluded']: continue
                progress['current']=name; self.store.job('running',progress,job_id)
                doc_id=old['id'] if old else str(uuid.uuid5(uuid.NAMESPACE_URL,str(path.resolve()).casefold()))
                try:
                    before=path.stat()
                    wait=min(self.settings.stable_seconds,max(0,self.settings.stable_seconds-(time.time()-before.st_mtime)))
                    if wait:time.sleep(wait)
                    after=path.stat()
                    if (before.st_mtime_ns,before.st_size)!=(after.st_mtime_ns,after.st_size): raise RuntimeError('File is still being saved; retry next scan')
                    content_hash=digest(path)
                    parser_fingerprint=parsing_fingerprint(self.settings)
                    cache_fingerprint=parser_fingerprint
                    if path.suffix.lower()=='.py':
                        parser_fingerprint=hashlib.sha256((parser_fingerprint+':python-ast-v3').encode()).hexdigest()
                        cache_fingerprint=parser_fingerprint
                    if path.suffix.lower()=='.pdf':
                        import copy
                        overrides=self.settings.pdf_overrides
                        selected=overrides.get(path.relative_to(self.settings.corpus).as_posix(),
                            overrides.get(path.name,self.settings.pdf_mode))
                        parser_fingerprint=hashlib.sha256((parser_fingerprint+json.dumps(selected,sort_keys=True)).encode()).hexdigest()
                        # Keep page checkpoints beside prior routes. Parsed-document
                        # provenance still binds the actual override configuration.
                        base=copy.copy(self.settings);base.pdf_overrides={}
                        cache_fingerprint=hashlib.sha256((parsing_fingerprint(base)+
                            json.dumps(base.pdf_mode,sort_keys=True)).encode()).hexdigest()
                    # The full pipeline fingerprint binds parser, chunker and model.
                    image_version=':visual-description-v1' if path.suffix.lower() in {'.png','.jpg','.jpeg','.webp','.bmp'} and self.models else ''
                    version=hashlib.sha256((content_hash+generation_fingerprint+parser_fingerprint+self.settings.collection+image_version).encode()).hexdigest()
                    if old is None:
                        same=[d for d in bypath.values() if d['hash']==content_hash and d['id'] not in renamed_ids]
                        moved=[d for d in same if d['path'] not in seen and d['version']==version]
                        # 只有唯一旧路径消失且内容/版本匹配时才认定改名，避免误合并重复文件。
                        if len(moved)==1:
                            previous=moved[0]
                            with self.store.db() as db:
                                db.execute('UPDATE documents SET path=?,status=?,updated=? WHERE id=?',
                                    (name,'excluded' if previous['excluded'] else 'ready',time.time(),previous['id']))
                                saved=list(db.execute('SELECT id,text,locator FROM chunks WHERE document_id=?',(previous['id'],)))
                                for c in saved:
                                    locator=json.loads(c['locator'])
                                    if 'image' in locator:
                                        locator['image']=path.name
                                        db.execute('UPDATE chunks SET locator=? WHERE id=?',(json.dumps(locator,ensure_ascii=False),c['id']))
                                db.execute('DELETE FROM chunks_fts WHERE id IN (SELECT id FROM chunks WHERE document_id=?)',(previous['id'],))
                                db.executemany('INSERT INTO chunks_fts VALUES(?,?)',[(c['id'],terms(path.name+' '+embedding_text(dict(c)))) for c in saved])
                            renamed_ids.add(previous['id']);progress['renamed']+=1
                            if not previous['excluded'] and previous['status']!='deleted':progress['deleted']-=1
                            continue
                        if same:progress['duplicates']+=1
                    if old and old['version']==version and old['status']=='ready':
                        progress['unchanged']+=1; continue
                    cache=self.settings.state/'cache'/content_hash/cache_fingerprint/path.suffix.lower().lstrip('.')
                    cache.mkdir(parents=True,exist_ok=True)
                    parsed_file=cache/'parsed.json'
                    provenance={'content_sha256':content_hash,'parser_fingerprint':parser_fingerprint,
                                'extension':path.suffix.lower()}
                    saved=cached_json(parsed_file);blocks=None
                    if isinstance(saved,dict) and saved.get('provenance')==provenance:
                        candidate=saved.get('blocks')
                        checksum=hashlib.sha256(json.dumps(candidate,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
                        if saved.get('blocks_sha256')==checksum:blocks=candidate
                    if blocks is None:
                        blocks=parse(path,cache,self.settings)
                        if digest(path)!=content_hash:
                            raise RuntimeError('File changed during parsing; parsed cache was not published')
                        save_json(parsed_file,{'provenance':provenance,'blocks':blocks,
                            'blocks_sha256':hashlib.sha256(json.dumps(blocks,ensure_ascii=False,sort_keys=True).encode()).hexdigest()})
                    if path.suffix.lower() in {'.png','.jpg','.jpeg','.webp','.bmp'}:
                        snapshot=cache/('original'+path.suffix.lower())
                        if not snapshot.exists():shutil.copy2(path,snapshot)
                        blocks=[{**item,'image':str(snapshot),'locator':{**item['locator'],'image':path.name}} for item in blocks]
                        if self.models:
                            caption_path=cache/'visual-description-v1.json'
                            caption=cached_json(caption_path)
                            if caption is None:
                                caption=self.models.vision(snapshot,'请客观描述图片的主体、颜色、布局、可读文字；图表说明坐标轴和图例。不要识别人名，不推测没有呈现的细节。',
                                    'production','preparation')
                                save_json(caption_path,caption)
                            blocks.append({'text':'[AI-generated visual description; verify original image for details; model='+str(caption['actual_model'])+']\n'+caption['text'],
                                'locator':{'image':path.name,'kind':'generated_description'},'image':str(snapshot)})
                    chunks=[]
                    for n,item in enumerate(self._chunks(blocks,tokenizer)):
                        item['retrieval_text']=self._retrieval_text(item,tokenizer)
                        item['locator']['retrieval_text']=item['retrieval_text']
                        item['locator']['parent_id']=hashlib.sha256((doc_id+item['locator']['parent_id']).encode()).hexdigest()[:24]
                        item['locator']['pipeline_fingerprint']=generation_fingerprint
                        chunks.append({**item,'id':str(uuid.uuid5(uuid.NAMESPACE_URL,doc_id+version+str(n))),
                                       'document_id':doc_id,'version':version})
                    if not chunks: raise ValueError('No extractable content; document not published')
                    # 向量先落盘；台账未切换之前，新块不会被 active() 暴露给查询。
                    self.retrieval.upsert(chunks)
                    if digest(path)!=content_hash: raise RuntimeError('File changed during indexing; old version retained')
                    with self.store.db() as db:
                        stale=[r[0] for r in db.execute('SELECT id FROM chunks WHERE document_id=?',(doc_id,))]
                        db.execute('DELETE FROM chunks_fts WHERE id IN (SELECT id FROM chunks WHERE document_id=?)',(doc_id,))
                        db.execute('DELETE FROM chunks WHERE document_id=?',(doc_id,))
                        db.executemany('INSERT INTO chunks VALUES(?,?,?,?,?,?,?)',[(c['id'],doc_id,version,c['text'],json.dumps(c['locator'],ensure_ascii=False),c['parent'],c['image']) for c in chunks])
                        db.executemany('INSERT INTO chunks_fts VALUES(?,?)',[(c['id'],terms(path.name+' '+embedding_text(c))) for c in chunks])
                        db.execute('''INSERT INTO documents VALUES(?,?,?,?,?,?,0,?) ON CONFLICT(path) DO UPDATE
                          SET hash=excluded.hash,version=excluded.version,status='ready',error=NULL,updated=excluded.updated''',
                          (doc_id,name,content_hash,version,'ready',None,time.time()))
                    # 台账提交后旧块已被版本过滤屏蔽，再清理旧向量，不让查询看到半次更新。
                    self.retrieval.remove([i for i in stale if i not in {c['id'] for c in chunks}])
                    progress['completed']+=1
                except Exception as exc:
                    error=f'{type(exc).__name__}: {exc}'[:1500]
                    with self.store.db() as db:
                        # 已有文档失败时只改状态和错误，不覆盖上次成功的 hash/version。
                        db.execute('''INSERT INTO documents VALUES(?,?,NULL,NULL,'update_failed',?,0,?)
                          ON CONFLICT(path) DO UPDATE SET status='update_failed',error=excluded.error,updated=excluded.updated''',
                          (doc_id,name,error,time.time()))
                    progress['failed']+=1; progress['errors'].append({'path':name,'error':error})
                self.store.job('running',progress,job_id)
            if hasattr(self.retrieval,'collect_garbage'):
                progress['stale_vectors_removed']=self.retrieval.collect_garbage()
            self.store.job('partial' if progress['failed'] else 'complete',progress,job_id)
            return {'job_id':job_id,**progress}
        except Exception as exc:
            self.store.job('failed',{**progress,'error':str(exc)},job_id); raise
        finally:
            self.lock.release()
            process_lock.release()
