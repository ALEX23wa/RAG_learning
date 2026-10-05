"""Isolated 100k-vector storage/restart benchmark; never a semantic accuracy test."""
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import time
import sys
import httpx
import numpy as np
import psutil
from qdrant_client import QdrantClient, models

ROOT=Path(__file__).resolve().parents[1]/'.rag'
FOLDER=ROOT/'scale'; FOLDER.mkdir(exist_ok=True)
URL='http://127.0.0.1:6335'

def start():
    """启动隔离的压力测试 Qdrant 进程并等待健康检查成功。

    Args:
        无显式输入参数。

    Returns:
        subprocess.Popen：已就绪的进程句柄，由调用方负责终止。

    Raises:
        RuntimeError：连续健康检查未通过。

    Notes:
        无显式参数；使用 .rag/scale 存储和 6335/6336 端口，避免写入真实知识库；日志追加到该目录。
    """
    config=FOLDER/'config.yaml'
    config.write_text('storage:\n  storage_path: '+json.dumps(str(FOLDER/'storage'))+
        '\n  on_disk_payload: true\nservice:\n  host: 127.0.0.1\n  http_port: 6335\n  grpc_port: 6336\ntelemetry_disabled: true\n',encoding='utf-8')
    with (FOLDER/'qdrant.log').open('ab') as log:
        p=subprocess.Popen([str(next((ROOT/'qdrant').rglob('qdrant.exe'))),'--config-path',str(config)],
            cwd=FOLDER,stdout=log,stderr=log,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    for _ in range(100):
        try:
            httpx.get(URL+'/healthz',timeout=1).raise_for_status(); return p
        except httpx.HTTPError: time.sleep(.2)
    raise RuntimeError('Scale database startup failed')

def main():
    """执行十万条合成向量的写入、删除、查询和重启恢复测试。

    Args:
        无显式输入参数。

    Returns:
        None：将状态、耗时（秒）、内存/磁盘字节数等写入 scale-100k.json 并打印。

    Notes:
        无显式参数；使用固定随机种子生成 768 维向量，重建隔离的 scale 集合并删去 1000 条。finally 终止测试数据库；这是存储测试，不是语义准确率评测。
    """
    report={'kind':'synthetic_storage_only','count':100000,'dimension':768,'status':'running'}
    process=start(); started=time.time(); peak=0
    try:
        client=QdrantClient(url=URL,timeout=180)
        if client.collection_exists('scale'): client.delete_collection('scale')
        client.create_collection('scale',vectors_config=models.VectorParams(size=768,distance=models.Distance.COSINE,on_disk=True),
            hnsw_config=models.HnswConfigDiff(on_disk=True),on_disk_payload=True)
        db=sqlite3.connect(FOLDER/'ledger.sqlite3'); db.execute('CREATE TABLE IF NOT EXISTS items(id INTEGER PRIMARY KEY, version INTEGER, text TEXT)')
        rng=np.random.default_rng(271828)
        for begin in range(0,100000,256):
            end=min(begin+256,100000); vectors=rng.normal(size=(end-begin,768)).astype('float32')
            vectors/=np.linalg.norm(vectors,axis=1,keepdims=True)
            client.upsert('scale',[models.PointStruct(id=i,vector=v.tolist(),payload={'version':1}) for i,v in zip(range(begin,end),vectors)],wait=True)
            db.executemany('INSERT OR REPLACE INTO items VALUES(?,1,?)',[(i,f'synthetic chunk {i}') for i in range(begin,end)]); db.commit()
            peak=max(peak,psutil.Process(process.pid).memory_info().rss+psutil.Process().memory_info().rss)
            if begin%4096==0: print(end,flush=True)
        report['write_seconds']=time.time()-started
        report['before_delete']=client.count('scale',exact=True).count
        client.delete('scale',models.PointIdsList(points=list(range(1000))),wait=True)
        report['after_delete']=client.count('scale',exact=True).count
        vector=vectors[0].tolist(); t=time.time()
        results=client.query_points('scale',query=vector,limit=10).points
        report['query_seconds']=time.time()-t
        report['query_ids']=[p.id for p in results]
        client.close(); process.terminate(); process.wait(timeout=30)
        process=start(); client=QdrantClient(url=URL,timeout=180)
        report['after_restart']=client.count('scale',exact=True).count
        report['deleted_not_returned']=not client.retrieve('scale',ids=list(range(1000)))
        report['peak_rss_bytes']=peak
        report['disk_bytes']=sum(p.stat().st_size for p in FOLDER.rglob('*') if p.is_file())
        report['status']='passed' if report['before_delete']==100000 and report['after_delete']==report['after_restart']==99000 and report['deleted_not_returned'] else 'failed'
        db.close(); client.close()
    except Exception as exc:
        report['status']='failed'; report['error']=str(exc)
    finally:
        if process.poll() is None: process.terminate(); process.wait(timeout=30)
        report['elapsed']=time.time()-started
        (ROOT/'reports'/'scale-100k.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__': main()
