import argparse
import json
from pathlib import Path
from .service import Service

def main():
    """解析命令行并分派服务启动、同步、问答、评测或维护操作。

    Args:
        无显式输入参数。

    Returns:
        None：命令结果以 JSON 输出到标准输出；serve 持续运行直到进程停止。

    Notes:
        无显式参数；从 sys.argv 读取子命令和选项。实际磁盘/网络/模型副作用由对应操作执行；参数错误由 argparse 退出。
    """
    parser=argparse.ArgumentParser(prog='python -m ragapp')
    parser.add_argument('--config',help='JSON Settings overrides; experiments should set load_active=false')
    parser.add_argument('--campaign',default='',help='Explicit independent paid-call campaign, e.g. quality-v2')
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ('sync','doctor','rebuild','manifest'): sub.add_parser(name)
    publish=sub.add_parser('publish');publish.add_argument('generation');publish.add_argument('report')
    serve=sub.add_parser('serve'); serve.add_argument('--port',type=int,default=8000); serve.add_argument('--no-sync',action='store_true')
    for name in ('search','ask'):
        p=sub.add_parser(name); p.add_argument('question')
        if name=='ask': p.add_argument('--profile',choices=['production','eval_aihub_free'],default='production')
    ev=sub.add_parser('evaluate'); ev.add_argument('cases'); ev.add_argument('report'); ev.add_argument('--profile',choices=['production','eval_aihub_free'],default='eval_aihub_free')
    ev.add_argument('--stage',choices=['development','acceptance'],default='acceptance')
    audit=sub.add_parser('audit');audit.add_argument('report');audit.add_argument('review');audit.add_argument('--expected',type=int,default=60)
    for name in ('backup','restore'):
        p=sub.add_parser(name); p.add_argument('archive')
    args=parser.parse_args()
    from .config import Settings
    values=json.loads(Path(args.config).read_text(encoding='utf-8')) if args.config else {}
    for key in ('root','corpus','state'):
        if key in values:values[key]=Path(values[key])
    if args.campaign:values['campaign']=args.campaign
    service=Service(Settings(**values))
    if args.command=='serve':
        import uvicorn
        from .api import create_app
        uvicorn.run(create_app(service,background_sync=not args.no_sync),host='127.0.0.1',port=args.port)
        return
    if args.command=='audit':
        from .audit import finalize
        result=finalize(args.report,args.review,args.expected)
    elif args.command=='rebuild':
        from .maintenance import rebuild
        result=rebuild(service)
    elif args.command=='manifest':result=service.manifest()
    elif args.command=='publish':
        from .maintenance import publish_generation
        result=publish_generation(service,args.generation,args.report)
    elif args.command in ('backup','restore'):
        from .maintenance import backup,restore
        result=(backup if args.command=='backup' else restore)(service,args.archive)
    elif args.command=='doctor':
        import sys,shutil
        from data_process.data_md import mineru_gpu_devices
        result={'python':sys.executable,'corpus':str(service.settings.corpus),'documents':len(service.indexer.scan()),
                'models':{name:(service.settings.state/'models'/name/'config.json').exists() for name in ('embedding','reranker')},
                'mineru_devices':mineru_gpu_devices(),'costs':service.costs()}
    elif args.command=='evaluate':
        from .evaluation import run
        tokens=json.loads((service.settings.state/'tokens.json').read_text())
        result=run(args.cases,args.report,tokens['eval_aihub_free' if args.profile=='eval_aihub_free' else ('admin' if args.stage=='development' else 'blackbox')],trace_token=tokens['admin'])
    else:
        service.start()
        if args.command=='sync': result=service.indexer.sync()
        elif args.command=='search': result=service.search(args.question)
        else: result=service.ask(args.question,args.profile)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
