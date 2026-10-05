from pathlib import Path
from types import SimpleNamespace
import pytest
from ragapp.config import Settings
from ragapp.store import Store
from ragapp.indexer import Indexer

class Tokenizer:
    def __call__(self,text,**kwargs):
        """用逐字符 offset 模拟 tokenizer，隔离真实模型加载。

        Args:
            text: str：待切分测试文本。
            kwargs: dict：兼容真实 tokenizer 的额外关键字参数；本模拟器忽略其值。

        Returns:
            dict：offset_mapping 为每个字符的左闭右开区间。
        """
        return {'offset_mapping':[(i,i+1) for i in range(len(text))]}

class Vectors:
    embedding=SimpleNamespace(tokenizer=Tokenizer())
    def __init__(self):
        """初始化只记录块 ID 的假向量存储。

        Args:
            无显式输入参数。

        Returns:
            None：ids 为空集合，fail 为 False。
        """
        self.ids=set();self.fail=False
    def upsert(self,chunks):
        """模拟向量写入或按测试要求注入失败。

        Args:
            chunks: Iterable[dict]：含 id 字段的测试块。

        Returns:
            None：成功时把输入块 ID 加入集合。

        Raises:
            RuntimeError：实例 fail 标志为 True。
        """
        if self.fail:raise RuntimeError('simulated vector write failure')
        self.ids.update(c['id'] for c in chunks)
    def remove(self,ids):
        """模拟删除已存储的向量 ID。

        Args:
            ids: Iterable[str]：待移除的测试块 ID。

        Returns:
            None：从实例 ids 集合移除匹配项，不存在的 ID 被忽略。
        """
        self.ids.difference_update(ids)

def fixture(tmp_path):
    """创建隔离的语料目录、SQLite 台账和假向量同步器。

    Args:
        tmp_path: Path：pytest 为当前测试提供的独立临时目录。

    Returns:
        tuple[Path, Store, Vectors, Indexer]：测试目录及相关组件；文件稳定等待时间设为 0。
    """
    papers=tmp_path/'papers';papers.mkdir()
    settings=Settings(root=tmp_path,corpus=papers,state=tmp_path/'state',stable_seconds=0).prepare()
    store=Store(settings.state/'catalog.sqlite3');vectors=Vectors()
    return papers,store,vectors,Indexer(settings,store,vectors)

def test_add_update_failure_recovery_rename_delete(tmp_path):
    """依次验证新增、更新失败保留旧版、恢复、跳过未变文件、改名复用 ID 和删除屏蔽。

    Args:
        tmp_path: Path：pytest 为当前测试提供的独立临时目录。

    Returns:
        None：使用断言判断测试结果；不满足条件时测试失败。
    """
    papers,store,vectors,index=fixture(tmp_path)
    source=papers/'a.txt';source.write_text('old evidence')
    assert index.sync()['completed']==1
    old=store.documents()[0];old_ids=list(vectors.ids)
    source.write_text('new evidence');vectors.fail=True
    assert index.sync()['failed']==1
    assert store.documents()[0]['version']==old['version']
    assert store.documents()[0]['status']=='update_failed'
    assert store.active(old_ids)
    vectors.fail=False
    assert index.sync()['completed']==1
    assert not store.active(old_ids)
    assert index.sync()['unchanged']==1
    source.rename(papers/'renamed.txt');index.sync()
    assert len(store.documents())==1 and store.documents()[0]['id']==old['id']
    assert any(d['path']=='renamed.txt' and d['status']=='ready' for d in store.documents())
    (papers/'renamed.txt').unlink();index.sync()
    assert not store.active(list(vectors.ids))

def test_exclusion_survives_sync_and_preserves_disk(tmp_path):
    """验证排除状态跨同步保留，磁盘原件不被删除。

    Args:
        tmp_path: Path：pytest 为当前测试提供的独立临时目录。

    Returns:
        None：使用断言判断测试结果；不满足条件时测试失败。
    """
    papers,store,vectors,index=fixture(tmp_path)
    source=papers/'a.txt';source.write_text('keep original')
    index.sync();store.exclude(store.documents()[0]['id']);index.sync()
    assert source.read_text()=='keep original'
    assert not store.active(list(vectors.ids))

def test_one_corrupt_file_does_not_block_others(tmp_path):
    """验证损坏 DOCX 不阻止同批正常文本入库。

    Args:
        tmp_path: Path：pytest 为当前测试提供的独立临时目录。

    Returns:
        None：使用断言判断测试结果；不满足条件时测试失败。
    """
    papers,store,vectors,index=fixture(tmp_path)
    (papers/'bad.docx').write_bytes(b'not a zip')
    (papers/'good.txt').write_text('good')
    result=index.sync()
    assert result['failed']==1 and result['completed']==1

def test_restarting_indexer_reuses_published_state(tmp_path):
    """验证重建索引器实例后复用已发布的版本台账。

    Args:
        tmp_path: Path：pytest 为当前测试提供的独立临时目录。

    Returns:
        None：使用断言判断测试结果；不满足条件时测试失败。
    """
    papers,store,vectors,index=fixture(tmp_path)
    (papers/'good.txt').write_text('good');index.sync()
    restarted=Indexer(index.settings,Store(store.path),vectors)
    assert restarted.sync()['unchanged']==1
