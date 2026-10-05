import asyncio
from contextlib import asynccontextmanager
import json
import secrets
import threading
from fastapi import FastAPI, Depends, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from .service import Service

class Question(BaseModel):
    """问答/检索输入；history 为 role/content 消息列表，服务只将用户问题用于指代。"""
    question: str = Field(min_length=1,max_length=8000)
    history: list[dict] = Field(default_factory=list,max_length=20)
    document_ids: list[str] | None = Field(default=None,max_length=100)
    debug: bool = False
    options: dict = Field(default_factory=dict)

class Calculation(BaseModel):
    """显式表格计算输入；document_id 指向台账，column 必须精确匹配表头。"""
    document_id:str
    operation:str
    column:str
    sheet:str|None=None
    filters:dict=Field(default_factory=dict)

def create_app(service=None, *, background_sync=True, ui=True):
    """创建带能力令牌验证、HTTP 路由和可选网页界面的应用。

    Args:
        service: Service | None：复用的服务实例；None 时创建默认服务。
        background_sync: bool：是否启动扫描、文件监听及每 60 秒核对，默认 True。
        ui: bool：是否将 Gradio 网页挂载到根路径，默认 True。

    Returns:
        FastAPI：可交给 uvicorn 运行的应用对象。

    Notes:
        首次创建随机本机令牌文件；后台线程和 Qdrant 连接在应用生命周期开始时启动。
    """
    service=service or Service()
    token_file=service.settings.state/'tokens.json'
    if not token_file.exists():
        token_file.write_text(json.dumps({role:secrets.token_urlsafe(32) for role in ('admin','blackbox','eval_aihub_free')}))
    tokens=json.loads(token_file.read_text())
    stop=threading.Event()

    def role(authorization: str=Header(default='')):
        """将 HTTP Bearer 令牌解析为服务端能力角色。

        Args:
            authorization: str：FastAPI 从 Authorization 请求头注入的值，格式为 Bearer <token>。

        Returns:
            str：admin、blackbox 或 eval_aihub_free 之一。

        Raises:
            HTTPException(401)：未提供令牌或令牌无效。
        """
        token=authorization.removeprefix('Bearer ')
        for name,value in tokens.items():
            if secrets.compare_digest(token,value): return name
        raise HTTPException(401,'Bearer token required')

    def admin(current=Depends(role)):
        """限制需要管理能力的接口访问。

        Args:
            current: str：role 依赖已经验证的调用者角色。

        Returns:
            str：验证通过时返回 admin。

        Raises:
            HTTPException(403)：角色不是 admin。
        """
        if current!='admin': raise HTTPException(403,'Administrative capability required')
        return current

    def sync_loop():
        """在后台周期性核对资料清单，直到收到停止信号。

        Args:
            无显式输入参数。

        Returns:
            None：停止事件触发后退出；每轮结果由 Indexer 持久化。

        Notes:
            无显式参数；捕获外层 service/stop。两轮起始间隔至少约 60 秒；全局异常另存为失败任务。
        """
        import time
        while not stop.is_set():
            started=time.monotonic()
            try: service.indexer.sync()
            except Exception as exc:
                service.store.job('failed',{'error':str(exc)})
            stop.wait(max(1,60-(time.monotonic()-started)))

    def watch_loop():
        """监听资料目录的文件变化并在防抖后触发同步。

        Args:
            无显式输入参数。

        Returns:
            None：停止事件、监听结束或目录监听错误时退出。

        Notes:
            无显式参数；捕获外层 service/stop。防抖 2000 毫秒；已有同步正在进行时忽略重复触发，OSError 写入失败任务。
        """
        from watchfiles import watch
        try:
            for changes in watch(service.settings.corpus,stop_event=stop,debounce=2000,step=500):
                if stop.is_set():break
                if any('extracted' not in str(path) for _,path in changes):
                    try:service.indexer.sync()
                    except RuntimeError:pass
        except OSError as exc:service.store.job('failed',{'watcher':str(exc)})

    @asynccontextmanager
    async def lifespan(app):
        """管理应用启动锁、Qdrant 连接和同步线程的生命周期。

        Args:
            app: FastAPI：框架传入的应用对象，当前实现无需直接读取它。

        Yields:
            None：yield 期间 FastAPI 接收请求；退出后设置后台停止信号并释放锁。

        Notes:
            启动时把未完成任务标记为 interrupted；server.lock 用于阻止服务运行期间执行恢复/重建。
        """
        from filelock import FileLock
        server_lock=FileLock(str(service.settings.state/'server.lock'),timeout=0)
        server_lock.acquire()
        service.start()
        with service.store.db() as db:
            db.execute("UPDATE jobs SET status='interrupted' WHERE status IN ('running','pending')")
        if background_sync:
            threading.Thread(target=sync_loop,daemon=True).start()
            threading.Thread(target=watch_loop,daemon=True).start()
        yield
        stop.set()
        server_lock.release()

    app=FastAPI(title='Papers RAG',lifespan=lifespan)
    app.state.service=service

    @app.get('/api/health')
    def health():
        """返回本机数据库连通性、模型名称和语料就绪数量。

        Args:
            无显式输入参数。

        Returns:
            dict：status、qdrant、embedding、reranker、profiles、sync_running、documents_ready。

        Notes:
            无显式参数；该健康接口不需要令牌。只检查 Qdrant 连通，不代表在线模型额度和语义准确率已通过。
        """
        import httpx
        try:database=httpx.get(service.settings.qdrant_url+'/healthz',timeout=2).is_success
        except httpx.HTTPError:database=False
        return {'status':'running' if database else 'degraded','qdrant':database,
                'embedding':'multilingual-e5-base','reranker':'bge-reranker-v2-m3',
                'profiles':['production','eval_aihub_free'],'sync_running':service.indexer.lock.locked(),
                'documents_ready':sum(d['status']=='ready' for d in service.store.documents())}

    @app.post('/api/search')
    def search(request:Question,current=Depends(role)):
        """处理独立检索请求并隐藏内部资产路径和父块。

        Args:
            request: Question：至少含 question 的 Pydantic 请求对象。
            current: str：已验证的调用角色；本接口允许三个有效角色。

        Returns:
            dict：evidence 列表，包含可公开给调用者的证据与来源定位。
        """
        from .context import public_evidence
        if (request.debug or request.options) and current!='admin':raise HTTPException(403,'Administrative diagnostics required')
        allowed={'rrf_k','lexical_candidates','dense_candidates','rerank_candidates','retrieval_mode','expand_context','query_planning'}
        if set(request.options)-allowed:raise HTTPException(422,'Unsupported retrieval option')
        for key in ('rrf_k','lexical_candidates','dense_candidates','rerank_candidates'):
            if key in request.options and (type(request.options[key]) is not int or not 1<=request.options[key]<=160):
                raise HTTPException(422,'Retrieval limits must be integers in 1..160')
        if request.options.get('retrieval_mode','hybrid_rerank') not in {'bm25','dense','hybrid','hybrid_rerank'}:
            raise HTTPException(422,'Unknown retrieval mode')
        for key in ('expand_context','query_planning'):
            if key in request.options and type(request.options[key]) is not bool:raise HTTPException(422,'Expected boolean option')
        result=service.search_detailed(request.question,request.history,document_ids=request.document_ids,
            profile='eval_aihub_free' if current=='eval_aihub_free' else 'production',
            stage='development' if current=='admin' else 'acceptance',options=request.options)
        response={'evidence':[public_evidence(e) for e in result['evidence']],'trace_id':result['trace_id']}
        if request.debug:response.update({k:v for k,v in result.items() if k!='evidence'})
        return response

    @app.get('/api/manifest')
    def manifest(current=Depends(role)):
        value=service.manifest()
        if current!='admin':return {'runtime_fingerprint':value['runtime_fingerprint'],
                                   'index_fingerprint':value['index_fingerprint'],
                                   'source_fingerprints':value['source_fingerprints']}
        return value

    @app.get('/api/traces/{trace_id}',dependencies=[Depends(admin)])
    def trace(trace_id:str):
        result=service.store.trace(trace_id)
        if result is None:raise HTTPException(404,'Trace not found')
        return result

    @app.post('/api/chat')
    def chat(request:Question,current=Depends(role)):
        """根据令牌角色选择固定模型配置并执行一次问答。

        Args:
            request: Question：question 与可选 history，经过长度限制校验。
            current: str：已验证角色；eval_aihub_free 使用免费配置，其他角色使用 production。

        Returns:
            dict：Service.ask 的回答、证据、引用、模型及用量结果。

        Raises:
            HTTPException(429)：分钟限流，附 Retry-After 秒数；HTTPException(503)：模型/处理异常，附 call_id。

        Notes:
            admin 调用计入 development，其余角色计入 acceptance；客户端不能指定第三方地址。
        """
        profile='eval_aihub_free' if current=='eval_aihub_free' else 'production'
        try:
            if request.debug or request.options:raise HTTPException(422,'Chat does not accept search diagnostics/options')
            return service.ask(request.question,profile,request.history,'acceptance' if current!='admin' else 'development',request.document_ids)
        except Exception as exc:
            if isinstance(exc,HTTPException):raise
            from .llm import RateLimited
            detail={'message':str(exc),'call_id':getattr(exc,'call_id',None)}
            if isinstance(exc,RateLimited):raise HTTPException(429,detail,headers={'Retry-After':str(int(exc.seconds)+1)}) from exc
            raise HTTPException(503,detail) from exc

    @app.post('/api/sync',dependencies=[Depends(admin)])
    def sync():
        """登记一次异步同步任务并立即返回任务 ID。

        Args:
            无显式输入参数。

        Returns:
            dict：job_id，可通过任务状态接口轮询。

        Raises:
            HTTPException(409)：当前线程内已有同步任务运行。

        Notes:
            无显式参数；管理权限由路由依赖验证，实际同步在新后台线程执行。
        """
        if service.indexer.lock.locked(): raise HTTPException(409,'Sync running')
        job=service.store.job()
        def work():
            """在后台执行已登记的同步任务。

            Args:
                无显式输入参数。

            Returns:
                None：任务结果写入台账，不通过线程返回。

            Notes:
                无显式参数；使用外层 job/service；Indexer 已持久化处理异常，因此此处不重复向 HTTP 请求抛出。
            """
            try: service.indexer.sync(job)
            except Exception: pass  # Indexer records the exception in persistent job state.
        threading.Thread(target=work,daemon=True).start()
        return {'job_id':job}

    @app.get('/api/jobs/{job_id}',dependencies=[Depends(admin)])
    def job(job_id:str):
        """读取指定同步任务的持久化状态。

        Args:
            job_id: str：持久化任务的唯一 ID。

        Returns:
            dict：任务记录，包括状态及 JSON 字符串形式的 progress。

        Raises:
            HTTPException(404)：任务 ID 不存在。
        """
        result=service.store.get_job(job_id)
        if not result: raise HTTPException(404,'Job not found')
        return result

    @app.get('/api/documents',dependencies=[Depends(admin)])
    def documents():
        """提供管理端完整文档状态列表。

        Args:
            无显式输入参数。

        Returns:
            list[dict]：包含 ready、update_failed、deleted、excluded 等状态的文档记录。
        """
        return service.store.documents()

    @app.get('/api/documents/{document_id}/source',dependencies=[Depends(admin)])
    def source(document_id:str,version:str|None=None):
        """校验文档状态和版本后提供原文件下载响应。

        Args:
            document_id: str：文档台账中的唯一 ID，不是磁盘文件名。
            version: str | None：查询参数中的引用版本；省略时使用当前版本。

        Returns:
            FileResponse：以原文件名作为附件名的下载响应。

        Raises:
            HTTPException(409)：原文不可用、过期或发生内容变更。
        """
        try: path=service.source(document_id,version)
        except ValueError as exc: raise HTTPException(409,str(exc)) from exc
        return FileResponse(path,filename=path.name,content_disposition_type='attachment')

    @app.get('/api/documents/{document_id}/preview',dependencies=[Depends(admin)])
    def preview(document_id:str):
        """校验原文后读取当前版本的前 20 个文本块用于预览。

        Args:
            document_id: str：文档台账中的唯一 ID，不是磁盘文件名。

        Returns:
            list[dict]：每项含 text、locator（JSON 字符串）和 version。

        Raises:
            HTTPException(409)：文档状态、路径或哈希校验失败。
        """
        try:service.source(document_id)
        except ValueError as exc:raise HTTPException(409,str(exc)) from exc
        with service.store.db() as db:
            rows=db.execute('''SELECT c.text,c.locator,c.version FROM chunks c JOIN documents d
              ON c.document_id=d.id AND c.version=d.version WHERE d.id=? LIMIT 20''',(document_id,))
            return [dict(r) for r in rows]

    @app.delete('/api/documents/{document_id}',dependencies=[Depends(admin)])
    def delete(document_id:str):
        """将指定资料移出检索范围并保留磁盘原文件。

        Args:
            document_id: str：文档台账中的唯一 ID，不是磁盘文件名。

        Returns:
            dict：excluded=True、original_preserved=True。

        Raises:
            HTTPException(404)：没有找到该文档 ID。
        """
        if not service.store.exclude(document_id): raise HTTPException(404,'Document not found')
        return {'excluded':True,'original_preserved':True}

    @app.get('/api/costs',dependencies=[Depends(admin)])
    def costs():
        """向管理端返回按模型配置、阶段和状态分组的调用统计。

        Args:
            无显式输入参数。

        Returns:
            list[dict]：Service.costs 返回的调用次数与人民币保守费用估算。
        """
        return service.costs()

    @app.post('/api/calculate',dependencies=[Depends(admin)])
    def calculate(request:Calculation):
        """校验原文版本后执行显式指定的表格聚合。

        Args:
            request: Calculation：document_id、operation、column，以及可选 sheet/filters。

        Returns:
            dict：计算值、参与行数、列/工作表、过滤条件及原表物理行范围。

        Raises:
            HTTPException(422)：文件版本失效、计算参数非法或数据无法安全聚合。
        """
        from .calculator import calculate as aggregate
        try:
            path=service.source(request.document_id)
            return aggregate(path,request.operation,request.column,sheet=request.sheet,filters=request.filters)
        except ValueError as exc:raise HTTPException(422,str(exc)) from exc

    if ui:
        import gradio as gr
        def answer(question,token,history):
            """处理网页提问并更新当前浏览器会话的多轮历史。

            Args:
                question: str：用户当前提出的自然语言问题。
                token: str：网页输入的本机 admin 能力令牌，使用常量时间比较校验。
                history: list[dict]：Gradio 会话保存的历史消息，role/content 格式。

            Returns:
                tuple[str, dict, list[dict]]：Markdown 回答、完整调用明细、更新后最多 12 条消息。

            Raises:
                gr.Error：管理令牌校验失败。
            """
            if not secrets.compare_digest(token,tokens['admin']): raise gr.Error('请输入本机管理令牌')
            result=service.ask(question,history=history)
            history=(history+[{'role':'user','content':question},{'role':'assistant','content':result['answer']}])[-12:]
            return result['answer'],result,history
        def ui_search(question,token):
            """处理网页独立检索按钮。

            Args:
                question: str：用户当前提出的自然语言问题。
                token: str：网页输入的本机 admin 能力令牌，使用常量时间比较校验。

            Returns:
                list[dict]：带来源定位和引用标识的证据列表。

            Raises:
                gr.Error：管理令牌校验失败。
            """
            if not secrets.compare_digest(token,tokens['admin']): raise gr.Error('请输入本机管理令牌')
            return service.search(question)
        def ui_status(token):
            """汇总网页所需的资料、费用和最近任务状态。

            Args:
                token: str：网页输入的本机 admin 能力令牌，使用常量时间比较校验。

            Returns:
                dict：documents、costs、sync_running，以及最近 5 条 jobs。

            Raises:
                gr.Error：管理令牌校验失败。
            """
            if not secrets.compare_digest(token,tokens['admin']): raise gr.Error('请输入本机管理令牌')
            with service.store.db() as db: jobs=[dict(r) for r in db.execute('SELECT * FROM jobs ORDER BY created DESC LIMIT 5')]
            return {'documents':service.store.documents(),'costs':service.costs(),'sync_running':service.indexer.lock.locked(),'jobs':jobs}
        def ui_sync(token):
            """验证网页管理令牌后发起异步同步。

            Args:
                token: str：网页输入的本机 admin 能力令牌，使用常量时间比较校验。

            Returns:
                dict：新任务 job_id。

            Raises:
                gr.Error：令牌错误；HTTPException(409)：已有同步运行。
            """
            if not secrets.compare_digest(token,tokens['admin']):raise gr.Error('请输入本机管理令牌')
            return sync()
        def ui_preview(document_id,token):
            """验证网页管理令牌并读取指定资料的原文片段。

            Args:
                document_id: str：文档台账中的唯一 ID，不是磁盘文件名。
                token: str：网页输入的本机 admin 能力令牌，使用常量时间比较校验。

            Returns:
                list[dict]：前 20 个当前版本文本块及定位。

            Raises:
                gr.Error：令牌错误；HTTPException(409)：原文件校验失败。
            """
            if not secrets.compare_digest(token,tokens['admin']):raise gr.Error('请输入本机管理令牌')
            return preview(document_id)
        with gr.Blocks(title='Papers 知识检索') as demo:
            gr.Markdown('# Papers 知识检索\n基于本地资料回答，并提供文件版本和证据定位。')
            token=gr.Textbox(label='本机管理令牌（.rag/tokens.json 中的 admin）',type='password')
            question=gr.Textbox(label='问题',lines=3)
            history=gr.State([])
            output=gr.Markdown(); detail=gr.JSON(label='证据、引用与调用记录')
            with gr.Row():
                ask=gr.Button('提问',variant='primary'); find=gr.Button('检索'); status=gr.Button('资料状态与用量'); synchronize=gr.Button('立即同步'); clear=gr.Button('清空对话上下文')
            document_id=gr.Textbox(label='资料 ID（从检索引用或资料状态复制）')
            preview_button=gr.Button('预览原文片段与定位')
            ask.click(answer,[question,token,history],[output,detail,history])
            clear.click(lambda:([],''),outputs=[history,output])
            find.click(ui_search,[question,token],detail)
            status.click(ui_status,[token],detail)
            synchronize.click(ui_sync,[token],detail)
            preview_button.click(ui_preview,[document_id,token],detail)
        app=gr.mount_gradio_app(app,demo,path='/')
    return app
