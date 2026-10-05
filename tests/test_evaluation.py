import json
import httpx
from ragapp.evaluation import run

def test_soft_quota_stops_at_same_case_without_paid_retry(tmp_path,monkeypatch):
    """验证 HTTP 200 包装的额度错误停在原题、保存断点且不切换付费请求。

    Args:
        tmp_path: Path：pytest 为当前测试提供的独立临时目录。
        monkeypatch: pytest.MonkeyPatch：替换网络或其他外部依赖，测试结束后自动恢复。

    Returns:
        None：使用断言判断测试结果；不满足条件时测试失败。
    """
    cases=tmp_path/'cases.json';report=tmp_path/'report.json'
    cases.write_text(json.dumps([{'id':'one','question':'q','answerable':True,'facts':[['ok']]},
                                {'id':'two','question':'q2','answerable':True,'facts':[['ok']]}]))
    requests=[]
    def post(self,url,**kw):
        """模拟 RAG API 以成功状态包裹免费额度错误的响应。

        Args:
            url: str：评测器请求的本机接口地址。
            kw: dict：包含 json 问题体的请求选项；记录请求以检查没有额外重试。

        Returns:
            httpx.Response：含 request 的 HTTP 200，用于验证 runner 自身的防御性检测。
        """
        requests.append(kw['json'])
        return httpx.Response(200,json={'answer':'Sorry, to prevent abuse of free resources, accounts that have not been recharged can only try 10 times.'},
                              request=httpx.Request('POST',url))
    monkeypatch.setattr(httpx.Client,'post',post)
    monkeypatch.setattr(httpx.Client,'get',lambda self,url,**kw: httpx.Response(200,
        json={'runtime_fingerprint':'stable','configuration':{},'index':{},'documents':{}},
        request=httpx.Request('GET',url)))
    run(cases,report,'test-only-token')
    result=json.loads(report.read_text())
    assert result['status']=='incomplete' and result['results']==[]
    assert result['blocked']['id']=='one' and result['metrics']['accuracy'] is None
    assert len(requests)==1
