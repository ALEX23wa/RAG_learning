import json
from ragapp.config import Settings
from ragapp.service import Service
from ragapp.calculator import infer

def test_sheet_row_focus_is_exact(tmp_path):
    """验证精确行定位不会把第 4 行混淆为第 14 或 40 行。

    Args:
        tmp_path: Path：pytest 为当前测试提供的独立临时目录。

    Returns:
        None：使用断言判断测试结果；不满足条件时测试失败。
    """
    s=Service(Settings(root=tmp_path,state=tmp_path/'state'))
    with s.store.db() as db:
        db.execute("INSERT INTO documents VALUES('d','inventory.xlsx','h','v','ready',NULL,0,0)")
        for number in (4,14,40):
            db.execute('INSERT INTO chunks VALUES(?,?,?,?,?,?,?)',(str(number),'d','v','same device',json.dumps({'sheet':'附表3-设备','range':f'A{number}:E{number}'}),'',None))
    ids,docs=s.store.focused('附表3第 4 行设备的价格是多少？')
    assert ids==['4'] and docs=={'d'}

def test_date_calculator_does_not_confuse_single_letter_headers(tmp_path):
    """验证日期/文件名不会误命中单字母列，并正确应用年份过滤。

    Args:
        tmp_path: Path：pytest 为当前测试提供的独立临时目录。

    Returns:
        None：使用断言判断测试结果；不满足条件时测试失败。
    """
    path=tmp_path/'data.csv';path.write_text('date,e,tp\n1995-01-01,1,2\n1996-01-01,2,4\n')
    assert infer('ERA5 data.csv 全部 tp 列的总和',path)['value']==6
    assert infer('1995年 tp 列平均值',path)['value']==2

def test_report_acronym_is_not_a_code_symbol_filter(tmp_path):
    """验证报告名称缩写只限定来源，不触发代码常量筛选。

    Args:
        tmp_path: Path：pytest 为当前测试提供的独立临时目录。

    Returns:
        None：使用断言判断测试结果；不满足条件时测试失败。
    """
    s=Service(Settings(root=tmp_path,state=tmp_path/'state'))
    with s.store.db() as db:
        db.execute("INSERT INTO documents VALUES('d','IPCC_report.pdf','h','v','ready',NULL,0,0)")
        db.execute("INSERT INTO chunks VALUES('c','d','v','IPCC chapter text','{}','',NULL)")
    focused,sources=s.store.focused('IPCC 的审稿编辑是谁？')
    assert focused==[] and sources=={'d'}

def test_source_rejects_changed_content(tmp_path):
    """验证原文件内容变化后拒绝提供旧版本引用对应的原文。

    Args:
        tmp_path: Path：pytest 为当前测试提供的独立临时目录。

    Returns:
        None：使用断言判断测试结果；不满足条件时测试失败。
    """
    from ragapp.indexer import digest
    import pytest
    corpus=tmp_path/'papers';corpus.mkdir();path=corpus/'a.txt';path.write_text('old')
    s=Service(Settings(root=tmp_path,corpus=corpus,state=tmp_path/'state'))
    with s.store.db() as db:db.execute('INSERT INTO documents VALUES(?,?,?,?,?,?,?,?)',('d','a.txt',digest(path),'v','ready',None,0,0))
    assert s.source('d','v')==path
    path.write_text('changed')
    with pytest.raises(ValueError,match='changed'):s.source('d','v')
