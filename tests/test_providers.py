import json
import httpx
import pytest
from ragapp.config import Settings
from ragapp.store import Store
from ragapp.llm import Models,ProviderUnavailable

def models(tmp_path):
    """构造只使用临时台账和虚假密钥的模型客户端。

    Args:
        tmp_path: Path：pytest 为当前测试提供的独立临时目录。

    Returns:
        Models：供测试 monkeypatch 网络调用的实例。
    """
    settings=Settings(root=tmp_path,state=tmp_path/'state').prepare()
    instance=Models(settings,Store(settings.state/'catalog.sqlite3'))
    instance.keys={'AIHUB_API_KEY':'fake','DEEPSEEK_API_KEY':'fake'}
    return instance


@pytest.mark.parametrize('effort',[None,'low','high','max'])
def test_thinking_is_explicit_and_all_completion_tokens_are_billed(tmp_path,monkeypatch,effort):
    instance=models(tmp_path);sent=[]
    def post(url,**kw):
        sent.append(kw['json'])
        return httpx.Response(200,json={'model':'actual-model',
            'usage':{'prompt_tokens':100,'completion_tokens':500,
                     'completion_tokens_details':{'reasoning_tokens':450}},
            'choices':[{'message':{'content':'final answer','reasoning_content':'private reasoning'}}]})
    monkeypatch.setattr(httpx,'post',post)
    result=instance.complete([{'role':'user','content':'test'}],reasoning_effort=effort,max_tokens=6144)
    assert sent[0]['thinking']=={'type':'enabled' if effort else 'disabled'}
    assert sent[0].get('reasoning_effort')==effort
    assert ('temperature' in sent[0])==(effort is None)
    assert result['cost_cny_upper_estimate']==pytest.approx((100*.3+500*1.2)*10/1_000_000)
    assert 'private reasoning' not in json.dumps(result)
    with instance.store.db() as db:
        row=db.execute('SELECT cost,usage FROM calls').fetchone()
        assert row['cost']==result['cost_cny_upper_estimate']
        assert json.loads(row['usage'])['completion_tokens']==500


@pytest.mark.parametrize('profile,effort',[('eval_aihub_free','low'),('production','invalid')])
def test_invalid_thinking_configuration_does_not_reserve_or_call(tmp_path,monkeypatch,profile,effort):
    instance=models(tmp_path)
    monkeypatch.setattr(httpx,'post',lambda *a,**kw:pytest.fail('must not send'))
    with pytest.raises(ValueError):instance.complete([],profile,reasoning_effort=effort)
    with instance.store.db() as db:assert db.execute('SELECT COUNT(*) FROM calls').fetchone()[0]==0

def test_free_quota_failure_has_no_paid_fallback(tmp_path,monkeypatch):
    """验证免费额度错误保留失败台账，且只请求允许的免费模型。

    Args:
        tmp_path: Path：pytest 为当前测试提供的独立临时目录。
        monkeypatch: pytest.MonkeyPatch：替换网络或其他外部依赖，测试结束后自动恢复。

    Returns:
        None：使用断言判断测试结果；不满足条件时测试失败。
    """
    instance=models(tmp_path);sent=[]
    def post(url,**kw):
        """记录免费模型请求并模拟账户额度 429。

        Args:
            url: str：被替换的 httpx.post 目标地址。
            kw: dict：待验证的 httpx 请求参数，包含 json 消息/模型等，不向真实网络发送。

        Returns:
            httpx.Response：带 Retry-After=120 的 429 响应。
        """
        sent.append((url,kw['json']['model']))
        return httpx.Response(429,headers={'Retry-After':'120'},text='quota exhausted')
    monkeypatch.setattr(httpx,'post',post)
    with pytest.raises(ProviderUnavailable,match='checkpoint'):instance.complete([{'role':'user','content':'hello'}],'eval_aihub_free')
    assert sent==[('https://aihubmix.com/v1/chat/completions','coding-glm-5.1-free')]
    with instance.store.db() as db:
        assert db.execute('SELECT status FROM calls').fetchone()[0]=='failed'

def test_actual_model_and_usage_preserved(tmp_path,monkeypatch):
    """验证请求别名与实际模型分别记录，并保存 usage 和免费费用。

    Args:
        tmp_path: Path：pytest 为当前测试提供的独立临时目录。
        monkeypatch: pytest.MonkeyPatch：替换网络或其他外部依赖，测试结束后自动恢复。

    Returns:
        None：使用断言判断测试结果；不满足条件时测试失败。
    """
    instance=models(tmp_path)
    def post(url,**kw):
        """模拟返回真实模型名和 token 用量的成功请求。

        Args:
            url: str：被替换的 httpx.post 目标地址。
            kw: dict：待验证的 httpx 请求参数，包含 json 消息/模型等，不向真实网络发送。

        Returns:
            httpx.Response：HTTP 200，含 actual-model、回答和 usage。
        """
        return httpx.Response(200,json={'model':'actual-model','choices':[{'message':{'content':'answer'}}],
        'usage':{'prompt_tokens':10,'completion_tokens':20}})
    monkeypatch.setattr(httpx,'post',post)
    result=instance.complete([{'role':'user','content':'hi'}],'eval_aihub_free')
    assert result['actual_model']=='actual-model' and result['requested_model']=='coding-glm-5.1-free'
    with instance.store.db() as db:
        row=db.execute('SELECT actual_model,cost,usage FROM calls').fetchone()
        assert row[0]=='actual-model' and row[1]==0 and json.loads(row[2])['completion_tokens']==20

def test_http_200_free_account_error_is_not_an_answer(tmp_path,monkeypatch):
    """验证 HTTP 200 正文中的账户额度提示被认定为失败而不是模型答案。

    Args:
        tmp_path: Path：pytest 为当前测试提供的独立临时目录。
        monkeypatch: pytest.MonkeyPatch：替换网络或其他外部依赖，测试结束后自动恢复。

    Returns:
        None：使用断言判断测试结果；不满足条件时测试失败。
    """
    instance=models(tmp_path);sent=[]
    def post(url,**kw):
        """模拟以 HTTP 200 返回账户额度提示的异常网关行为。

        Args:
            url: str：被替换的 httpx.post 目标地址。
            kw: dict：待验证的 httpx 请求参数，包含 json 消息/模型等，不向真实网络发送。

        Returns:
            httpx.Response：正文为限额提示并含零 token 用量。
        """
        sent.append(kw['json']['model'])
        return httpx.Response(200,json={'model':'glm-5.3','choices':[{'message':{'content':
            'Sorry, to prevent abuse of free resources, accounts that have not been recharged can only try 10 times. You can increase the free quota after recharging; https://console.aihubmix.com/topup'}}],
            'usage':{'prompt_tokens':0,'completion_tokens':0}})
    monkeypatch.setattr(httpx,'post',post)
    with pytest.raises(ProviderUnavailable,match='Account free quota exhausted'):
        instance.complete([{'role':'user','content':'question'}],'eval_aihub_free')
    assert sent==['coding-glm-5.1-free']
    with instance.store.db() as db:
        row=db.execute('SELECT status,actual_model,cost FROM calls').fetchone()
        assert tuple(row)==('failed','glm-5.3',0)

def test_free_vision_error_cannot_be_reported_as_verified(tmp_path,monkeypatch):
    """验证免费视觉模型确实接收 image_url，图片不支持错误不能算作视觉复核成功。

    Args:
        tmp_path: Path：pytest 为当前测试提供的独立临时目录。
        monkeypatch: pytest.MonkeyPatch：替换网络或其他外部依赖，测试结束后自动恢复。

    Returns:
        None：使用断言判断测试结果；不满足条件时测试失败。
    """
    from PIL import Image
    instance=models(tmp_path);p=tmp_path/'image.png';Image.new('RGB',(10,10),'red').save(p)
    def post(url,**kw):
        """断言免费视觉模型和图片输入字段正确，再模拟不支持图片。

        Args:
            url: str：被替换的 httpx.post 目标地址。
            kw: dict：待验证的 httpx 请求参数，包含 json 消息/模型等，不向真实网络发送。

        Returns:
            httpx.Response：HTTP 400，内容为 images unsupported。
        """
        assert kw['json']['model']=='agents-a1-free'
        assert kw['json']['messages'][0]['content'][1]['type']=='image_url'
        return httpx.Response(400,text='images unsupported')
    monkeypatch.setattr(httpx,'post',post)
    with pytest.raises(ProviderUnavailable,match='unsupported'):instance.vision(p,'color?','eval_aihub_free','preparation')
