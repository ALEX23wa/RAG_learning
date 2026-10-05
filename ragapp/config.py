from dataclasses import dataclass, field
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]

@dataclass
class Settings:
    """本机运行配置；token 参数使用模型 tokenizer 单位，stable_seconds 使用秒。"""

    # root 用于读取项目配置；corpus 只放原始资料；state 放索引、缓存和模型。
    root: Path = ROOT
    corpus: Path = field(default_factory=lambda: ROOT / 'papers')
    state: Path = field(default_factory=lambda: ROOT / '.rag')
    qdrant_url: str = 'http://127.0.0.1:6333'
    collection: str = 'papers_e5_v1'
    catalog: str = 'catalog.sqlite3'
    chunk_tokens: int = 384
    overlap: int = 48
    embedding_batch: int = 12
    stable_seconds: float = 2.0
    index_schema: str = 'quality-v2'
    index_profile: str = 'quality-v2'
    pdf_mode: str = 'auto'
    pdf_overrides: dict = field(default_factory=dict)
    embedding_model: str = 'intfloat/multilingual-e5-base'
    embedding_revision: str = 'd128750597153bb5987e10b1c3493a34e5a4502a'
    embedding_directory: str = 'embedding'
    embedding_dimension: int = 768
    embedding_input_tokens: int = 480
    embedding_max_tokens: int = 512
    # Legacy keeps the historic model loading and fingerprint byte-for-byte.
    embedding_precision: str = 'legacy'
    reranker_model: str = 'BAAI/bge-reranker-v2-m3'
    reranker_revision: str = '953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e'
    reranker_directory: str = 'reranker'
    # Preserve the deployed retrieval budget until a measured candidate is published.
    # The opt-in quality experiment is config/quality-v2.json.
    reranker_max_tokens: int = 512
    lexical_candidates: int = 40
    dense_candidates: int = 40
    dense_fetch_limit: int = 1280
    rerank_candidates: int = 20
    evidence_limit: int = 10
    rrf_k: int = 10
    context_tokens: int = 8000
    generation_context_tokens: int = 32000
    query_tokens: int = 256
    retrieval_mode: str = 'hybrid_rerank'
    expand_context: bool = False
    query_planning: bool = False
    verify_answers: bool = False
    # Opt-in experimental judge mode; generation and baseline defaults stay unchanged.
    judge_reasoning_effort: str = 'disabled'
    judge_max_tokens: int = 6144
    judge_reasoning_scope: str = 'all'
    use_evidence_outline: bool = False
    campaign: str = ''
    # An explicit campaign is a separate authorization; historical calls remain intact.
    campaign_limit: float = 100.0
    campaign_stages: dict = field(default_factory=lambda: {'preparation':20.,'development':50.,'acceptance':30.})
    load_active: bool = True

    def prepare(self):
        """应用当前索引版本指针并创建运行所需目录。

        Args:
            无显式输入参数。

        Returns:
            Settings：当前实例自身，便于链式调用。

        Notes:
            读取 state/active.json 的索引与已验收 runtime_settings；不从指针继承费用授权或目录路径。
            没有 runtime_settings 的旧指针保留调用方设置和基线默认值。实验可用 load_active=False 显式绕过指针。
            创建受管目录，并仅在环境变量未设置时配置 Hugging Face 缓存。
        """
        import json
        active=self.state/'active.json'
        if self.load_active and active.exists():
            pointer=json.loads(active.read_text())
            self.collection=pointer['collection']
            self.catalog=pointer.get('catalog','catalog.sqlite3')
            # A legacy pointer must not inherit an FP32 candidate constructor.
            self.embedding_precision=pointer.get('settings',{}).get('embedding_precision','legacy')
            for key,value in pointer.get('settings',{}).items():
                if key in INDEX_SETTINGS: setattr(self,key,value)
            for key,value in validated_runtime_settings(pointer.get('runtime_settings',{})).items():
                setattr(self,key,value)
        if self.embedding_precision not in ('legacy','float32'):
            raise ValueError('embedding_precision must be legacy or float32')
        validated_runtime_settings({'judge_reasoning_effort':self.judge_reasoning_effort,
                                    'judge_max_tokens':self.judge_max_tokens,
                                    'judge_reasoning_scope':self.judge_reasoning_scope,
                                    'use_evidence_outline':self.use_evidence_outline})
        for folder in ('models', 'cache', 'reports', 'evaluation', 'logs'):
            (self.state / folder).mkdir(parents=True, exist_ok=True)
        os.environ.setdefault('HF_HOME', str(self.state / 'huggingface'))
        os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
        return self

INDEX_SETTINGS = {'index_schema','index_profile','pdf_mode','pdf_overrides','chunk_tokens','overlap',
    'embedding_model','embedding_revision','embedding_directory','embedding_dimension',
    'embedding_input_tokens','embedding_max_tokens','embedding_precision'}

# Only settings that determine retrieval, generation evidence and model resource
# use may travel with an accepted generation. Paths, endpoint addresses, campaign
# names and monetary authorizations remain local/operator controlled.
RUNTIME_SETTINGS = frozenset({'embedding_batch','reranker_model','reranker_revision','reranker_directory',
    'reranker_max_tokens','lexical_candidates','dense_candidates','dense_fetch_limit','rerank_candidates',
    'evidence_limit','rrf_k','context_tokens','generation_context_tokens','query_tokens','retrieval_mode',
    'expand_context','query_planning','verify_answers','judge_reasoning_effort','judge_max_tokens',
    'judge_reasoning_scope','use_evidence_outline'})


def validated_runtime_settings(values, *, complete=False):
    """Validate the persisted allowlist, without applying or authorizing a campaign.

    ``complete`` is required when producing a new published pointer. Missing keys
    are tolerated only while reading legacy/partial local pointers. Model folders
    are managed directory names under state/models, never arbitrary paths.
    """
    import re
    from pathlib import PureWindowsPath
    if not isinstance(values,dict) or not set(values)<=RUNTIME_SETTINGS:
        raise ValueError('Unsupported published runtime setting')
    if complete and set(values)!=RUNTIME_SETTINGS:
        raise ValueError('Published runtime settings must be complete')
    defaults=Settings()
    for key,value in values.items():
        if type(value) is not type(getattr(defaults,key)):
            raise ValueError('Invalid published runtime setting type: '+key)
        if type(value) is int and value<(0 if key=='rrf_k' else 1):
            raise ValueError('Invalid published runtime setting value: '+key)
        if type(value) is str and not value.strip():
            raise ValueError('Empty published runtime setting: '+key)
    if 'retrieval_mode' in values and values['retrieval_mode'] not in {'bm25','dense','hybrid','hybrid_rerank'}:
        raise ValueError('Unsupported published retrieval mode')
    if 'judge_reasoning_effort' in values and values['judge_reasoning_effort'] not in {'disabled','low','high','max'}:
        raise ValueError('Unsupported judge reasoning effort')
    if 'judge_reasoning_scope' in values and values['judge_reasoning_scope'] not in {'all','sufficiency'}:
        raise ValueError('Unsupported judge reasoning scope')
    if 'judge_max_tokens' in values and not 1024<=values['judge_max_tokens']<=16384:
        raise ValueError('Judge output budget must be between1024 and16384 tokens')
    if 'reranker_revision' in values and not re.fullmatch('[0-9a-f]{40}',values['reranker_revision']):
        raise ValueError('Published reranker revision must be an immutable commit')
    if 'reranker_directory' in values:
        name=values['reranker_directory']
        if name in {'.','..'} or '/' in name or '\\' in name or ':' in name or PureWindowsPath(name).is_absolute():
            raise ValueError('Published reranker directory must be a managed model folder name')
    return dict(values)

PROFILES = {
    # key 保存环境变量名而非密钥值；模型地址与用途由服务端固定。
    'production': {'url': 'https://api.deepseek.com', 'key': 'DEEPSEEK_API_KEY',
                   'model': 'deepseek-flash', 'vision': 'deepseek-flash', 'free': False},
    'eval_aihub_free': {'url': 'https://aihubmix.com/v1', 'key': 'AIHUB_API_KEY',
                        'model': 'coding-glm-5.1-free', 'vision': 'agents-a1-free', 'free': True},
}
