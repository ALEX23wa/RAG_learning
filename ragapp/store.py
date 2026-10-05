import contextlib
import json
import sqlite3
import time
import uuid
import re

def terms(text):
    """将文本转换为适合 SQLite FTS5 匹配的中文搜索词。

    Args:
        text: str：待索引的正文、文件名或查询文本。

    Returns:
        str：小写、去除纯标点后以空格连接的分词结果。
    """
    import jieba
    tokens = jieba.cut_for_search(text.lower())
    return ' '.join(t for t in tokens if re.search(r'\w', t))

class Store:
    def __init__(self, path):
        """初始化数据库路径，并创建尚不存在的表和 FTS5 索引。

        Args:
            path: Path | str：SQLite 文件路径；其父目录需已存在。

        Returns:
            None：初始化实例并写入数据库结构。

        Notes:
            启用 WAL；保留已有文档、任务和费用记录。
        """
        self.path = path
        with self.db() as db:
            # documents.version 是查询可见版本；chunks 可写好后再随台账事务发布。
            # calls 的 reserved/cost 单位为元，created/updated 为 Unix 秒。
            db.executescript('''
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS documents (
              id TEXT PRIMARY KEY, path TEXT UNIQUE, hash TEXT, version TEXT,
              status TEXT, error TEXT, excluded INTEGER DEFAULT 0, updated REAL);
            CREATE TABLE IF NOT EXISTS chunks (
              id TEXT PRIMARY KEY, document_id TEXT, version TEXT, text TEXT,
              locator TEXT, parent TEXT, image TEXT);
            CREATE INDEX IF NOT EXISTS chunk_version ON chunks(document_id,version);
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(id UNINDEXED, tokens);
            CREATE TABLE IF NOT EXISTS jobs (
              id TEXT PRIMARY KEY, status TEXT, progress TEXT, created REAL, updated REAL);
            CREATE TABLE IF NOT EXISTS calls (
              id TEXT PRIMARY KEY, profile TEXT, stage TEXT, requested_model TEXT,
              actual_model TEXT, status TEXT, reserved REAL, cost REAL, usage TEXT,
              error TEXT, created REAL, elapsed REAL);
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT);
            CREATE TABLE IF NOT EXISTS traces (id TEXT PRIMARY KEY, created REAL, payload TEXT);
            CREATE TABLE IF NOT EXISTS call_campaigns (call_id TEXT PRIMARY KEY, campaign TEXT NOT NULL);
            ''')

    @contextlib.contextmanager
    def db(self):
        """提供带提交、回滚和关闭处理的数据库上下文。

        Args:
            无显式输入参数。

        Yields:
            sqlite3.Connection：使用 sqlite3.Row 的连接，应通过 with 调用。

        Notes:
            正常退出提交事务；发生任意异常时回滚并重新抛出，最后始终关闭连接。连接等待锁的上限为 60 秒。
        """
        db = sqlite3.connect(self.path, timeout=60)
        db.row_factory = sqlite3.Row
        try:
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def documents(self):
        """读取完整文档台账，供状态显示及同步比对。

        Args:
            无显式输入参数。

        Returns:
            list[dict]：按相对路径排序的文档记录，包含已删除、排除和更新失败状态。
        """
        with self.db() as db:
            return [dict(r) for r in db.execute('SELECT * FROM documents ORDER BY path')]

    def active(self, ids):
        """从候选文本块中筛出当前仍可用于检索的版本。

        Args:
            ids: Sequence[str]：候选文本块 ID。

        Returns:
            dict[str, dict]：块 ID 到块内容及文档路径/状态的映射；空输入返回空字典。

        Notes:
            仅保留文档当前版本且未排除的块；update_failed 可继续使用已发布的旧版本。
        """
        if not ids:
            return {}
        with self.db() as db:
            rows = db.execute(f'''SELECT c.*,d.path,d.status FROM chunks c JOIN documents d
              ON c.document_id=d.id AND c.version=d.version
              WHERE d.excluded=0 AND d.status IN ('ready','update_failed')
              AND c.id IN ({','.join('?' for _ in ids)})''', ids)
            return {r['id']: dict(r) for r in rows}

    def lexical(self, query, limit=40, document_ids=None):
        """使用 FTS5/BM25 检索当前版本的关键词证据。

        Args:
            query: str：检索用的自然语言问题或关键词。
            limit: int：最大召回数，默认 40。
            document_ids: Collection[str] | None：可选文档 ID 范围；None 或空集合不限制来源。

        Returns:
            list[str]：按 BM25 排序的块 ID，数量不超过 limit。

        Notes:
            分词后最多取 80 个去重词，以 OR 查询；来源过滤在 LIMIT 之前执行。
        """
        if document_ids is not None and not document_ids: return []
        words = list(dict.fromkeys(terms(query).split()))[:80]
        if not words:
            return []
        expression = ' OR '.join('"' + t.replace('"', '""') + '"' for t in words)
        restriction=''
        parameters=[expression]
        if document_ids:
            restriction=' AND d.id IN ('+','.join('?' for _ in document_ids)+')'
            parameters.extend(document_ids)
        parameters.append(limit)
        with self.db() as db:
            rows = db.execute('''SELECT f.id FROM chunks_fts f JOIN chunks c ON c.id=f.id
              JOIN documents d ON d.id=c.document_id AND d.version=c.version
              WHERE chunks_fts MATCH ? AND d.excluded=0 AND d.status IN ('ready','update_failed')
              '''+restriction+' ORDER BY bm25(chunks_fts) LIMIT ?', parameters)
            return [r[0] for r in rows]

    def source_scope(self, query, document_ids=None):
        """Only explicit IDs, full filenames/paths or an exact sheet name restrict retrieval."""
        from pathlib import PureWindowsPath
        docs=[d for d in self.documents() if not d['excluded'] and d['status'] in {'ready','update_failed'}]
        if document_ids is not None:
            return {d['id'] for d in docs if d['id'] in document_ids}
        normalized=query.replace('\\','/').lower()
        paths=[d for d in docs if '/' in d['path'].replace('\\','/') and
               d['path'].replace('\\','/').lower() in normalized]
        matches=paths or [d for d in docs if re.search(r'(?<![A-Za-z0-9_.])'+re.escape(PureWindowsPath(d['path']).name.lower())+r'(?![A-Za-z0-9_.])',normalized)]
        if '根目录' in query and '课程' not in query:
            matches=[d for d in matches if '/' not in d['path'].replace('\\','/')]
        if ('子目录' in query or '课程作业目录' in query) and '根目录' not in query:
            matches=[d for d in matches if '/' in d['path'].replace('\\','/')]
        if matches:return {d['id'] for d in matches}
        sheets=re.findall(r'附表\s*(\d+(?:-\d+)?)',query)
        if sheets:
            with self.db() as db:
                rows=db.execute('''SELECT DISTINCT c.document_id,json_extract(c.locator,'$.sheet') AS sheet
                    FROM chunks c JOIN documents d ON c.document_id=d.id AND c.version=d.version
                    WHERE d.excluded=0 AND d.status IN ('ready','update_failed')''')
                return {r['document_id'] for r in rows if any(re.match(r'附表'+re.escape(s)+r'(?:-|：|\s|$)',r['sheet'] or '') for s in sheets)}
        if re.search(r'\b[A-Za-z0-9_.-]+\.(?:pdf|docx|xlsx|csv|md|txt|py)\b',query,re.I):return set()
        return None

    def save_trace(self, payload):
        trace_id=str(uuid.uuid4())
        with self.db() as db:
            db.execute('INSERT INTO traces VALUES(?,?,?)',(trace_id,time.time(),json.dumps(payload,ensure_ascii=False)))
        return trace_id

    def trace(self, trace_id):
        with self.db() as db: row=db.execute('SELECT payload FROM traces WHERE id=?',(trace_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def siblings(self, item):
        """Return active chunks from the same parent, never from another document/version."""
        locator=item['locator'] if isinstance(item['locator'],dict) else json.loads(item['locator'])
        parent_id=locator.get('parent_id')
        if not parent_id:return []
        with self.db() as db:
            rows=db.execute('''SELECT c.id FROM chunks c JOIN documents d ON d.id=c.document_id
                AND d.version=c.version WHERE c.document_id=? AND c.version=? AND d.excluded=0
                AND d.status IN ('ready','update_failed') AND json_extract(c.locator,'$.parent_id')=?''',
                (item['document_id'],item['version'],parent_id))
            ids=[r[0] for r in rows]
        return list(self.active(ids).values())

    def source_hints(self, query):
        """Resolve named Chinese document cues to advisory paths, never a hard scope.

        A suffix before a document noun can match a filename stem (for example
        '另一份概述课件' -> '*概述.pdf'). Keep all strongest matches, including
        ambiguity, and require an explicit document noun rather than a topic word.
        """
        from pathlib import PureWindowsPath
        docs=[d for d in self.documents() if not d['excluded'] and d['status'] in {'ready','update_failed'}]
        result=[];seen=set()
        for match in re.finditer(r'([\w\u4e00-\u9fff]{2,32})(课件|文档|报告|手册|笔记|论文)',query):
            prefix=match[1];matches=[]
            for doc in docs:
                stem=PureWindowsPath(doc['path']).stem.casefold()
                for length in range(len(prefix),1,-1):
                    hint=prefix[-length:]
                    if hint.casefold() in stem:
                        matches.append((length,hint,doc));break
            strongest=max((length for length,_,_ in matches),default=0)
            for length,hint,doc in matches:
                key=(match.group(),doc['id'])
                if length==strongest and key not in seen:
                    result.append({'phrase':match.group(),'matched_text':hint,
                        'document_id':doc['id'],'path':doc['path'],'kind':'inferred_preference'})
                    seen.add(key)
        return result

    def focused(self,query):
        """解析问题中明确指定的文件、日期、工作表行或代码常量。

        Args:
            query: str：检索用的自然语言问题或关键词。

        Returns:
            tuple[list[str], set[str] | None]：最多 40 个优先块 ID 与来源文档 ID 集合；未识别文件时为 ([], None)。

        Notes:
            代码定义优先于使用位置；根目录与课程子目录按问题区分，PDF 等非代码文件不使用常量过滤。
        """
        from pathlib import PureWindowsPath
        docs=self.documents()
        matches=[d for d in docs if PureWindowsPath(d['path']).name.lower() in query.lower() and not d['excluded']]
        if not matches:
            # Partial Latin product/report identifiers resolve names such as IPCC_AR6_... .
            for doc in docs:
                stem=PureWindowsPath(doc['path']).stem
                identifiers=re.findall(r'[A-Za-z][A-Za-z0-9]{3,}',stem)
                if not doc['excluded'] and any(identifier.lower() in query.lower() for identifier in identifiers):matches.append(doc)
        hinted={hint['document_id'] for hint in self.source_hints(query)}
        existing={doc['id'] for doc in matches}
        matches.extend(doc for doc in docs if doc['id'] in hinted-existing)
        sheets=re.findall(r'附表\s*(\d+(?:-\d+)?)',query)
        if not matches and sheets:
            matches=[d for d in docs if d['path'].lower().endswith(('.xlsx','.xls')) and not d['excluded']]
        if '根目录' in query and '课程' not in query: matches=[d for d in matches if '\\' not in d['path'] and '/' not in d['path']]
        if ('子目录' in query or '课程作业目录' in query) and '根目录' not in query: matches=[d for d in matches if '\\' in d['path'] or '/' in d['path']]
        if not matches:return [],None
        doc_ids=[d['id'] for d in matches]; ids=[];definitions=[]
        dates=re.findall(r'\b\d{4}-\d{2}-\d{2}\b',query)
        symbols=re.findall(r'\b[A-Z][A-Z_]{3,}\b',query)
        if not all(PureWindowsPath(d['path']).suffix.lower() in {'.py','.java','.cpp','.c','.h'} for d in matches):symbols=[]
        rows=re.findall(r'第\s*(\d+)\s*行',query)
        with self.db() as db:
            clauses=' OR '.join('c.document_id=?' for _ in doc_ids)
            candidates=db.execute(f'''SELECT c.id,c.text,c.locator FROM chunks c JOIN documents d ON d.id=c.document_id
                 AND d.version=c.version WHERE ({clauses}) AND d.status IN ('ready','update_failed')''',doc_ids)
            for c in candidates:
                locator=json.loads(c['locator'])
                locations=locator.get('source_spans') or [locator]
                if symbols and any(re.search(r'(?m)^\s*'+re.escape(symbol)+r'\s*(?::[^=\n]+)?=(?!=)',c['text']) for symbol in symbols):definitions.append(c['id'])
                if sheets and not any(re.match(r'附表'+re.escape(s)+r'(?:-|：|\s|$)',loc.get('sheet','')) for s in sheets for loc in locations):continue
                if dates and any(date in c['text'] for date in dates):ids.append(c['id'])
                elif symbols and any(symbol in c['text'] for symbol in symbols):ids.append(c['id'])
                elif rows and any(loc.get('row')==int(row) or re.search(rf'[A-Z]+{row}:',loc.get('range','')) for row in rows for loc in locations):ids.append(c['id'])
                elif locator.get('image'):ids.append(c['id'])
        return (definitions or ids)[:40],set(doc_ids)

    def job(self, status='pending', progress=None, job_id=None):
        """创建同步任务或覆盖已有任务的状态与进度。

        Args:
            status: str：任务状态，如 pending、running、partial、complete 或 failed。
            progress: dict | None：可 JSON 序列化的进度、计数或错误信息。
            job_id: str | None：已有任务 ID；省略时生成 UUID。

        Returns:
            str：新建或复用的任务 ID。

        Notes:
            时间戳采用 Unix 秒；更新记录时保留原创建时间。
        """
        job_id = job_id or str(uuid.uuid4())
        with self.db() as db:
            db.execute('''INSERT INTO jobs VALUES(?,?,?,?,?) ON CONFLICT(id) DO UPDATE
              SET status=excluded.status,progress=excluded.progress,updated=excluded.updated''',
              (job_id, status, json.dumps(progress or {}, ensure_ascii=False), time.time(), time.time()))
        return job_id

    def get_job(self, job_id):
        """按 ID 读取一个持久化任务。

        Args:
            job_id: str：持久化任务的唯一 ID。

        Returns:
            dict | None：任务记录；不存在时返回 None，progress 字段仍为 JSON 字符串。
        """
        with self.db() as db:
            row = db.execute('SELECT * FROM jobs WHERE id=?', (job_id,)).fetchone()
            return dict(row) if row else None

    def exclude(self, document_id):
        """将文档持久排除出知识库，保留磁盘原件。

        Args:
            document_id: str：文档台账中的唯一 ID，不是磁盘文件名。

        Returns:
            bool：是否更新到至少一条文档记录。

        Notes:
            当前检索立即受状态过滤影响；后续同步不会自动重新启用该文档。
        """
        with self.db() as db:
            changed = db.execute("UPDATE documents SET excluded=1,status='excluded' WHERE id=?", (document_id,)).rowcount
        return bool(changed)
