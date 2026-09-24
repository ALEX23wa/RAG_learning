# LangChain 1.x 中文接口使用手册

> 面向会 Python、正在学习 RAG 的开发者。核验日期：2026-09-23。版本基线：本项目实际安装的 `langchain==1.4.1`、`langchain-core==1.6.3`。本文覆盖核心与常用生态的公开接口，不是所有第三方集成的逐项全集。

## 阅读约定与快速导航

先读第 1—2 章理解包和输入输出，再按需求查询。第一次实践可直接运行第 9 章。表格中的签名均为**常用参数简写**，不是完整函数签名；`…` 表示省略可选参数。除明确标注“完整程序”外，代码均为局部示例，只在声明的前置条件下运行。

- [1. 包、版本与安装](#packages)
- [2. 统一调用接口与 Runnable](#runnable)
- [3. 模型与消息](#models)
- [4. 提示词与输出解析](#prompts)
- [5. 工具与 Agent](#agents)
- [6. 中间件、状态与会话](#memory)
- [7. RAG：加载、切分、向量化与检索](#rag)
- [8. 缓存、回调、追踪与排错](#operations)
- [9. 可复制的完整示例](#examples)
- [10. 旧版迁移与扩展索引](#migration)
- [11. 核验记录与适用边界](#verification)

| 你要做的事 | 首先查找的接口 | 返回什么 |
|---|---|---|
| 向模型提问 | `init_chat_model`、`ChatOpenAI.invoke` | 消息对象，通常为 `AIMessage` |
| 拼装提示词 | `ChatPromptTemplate.from_messages`、`.invoke` | `ChatPromptValue` |
| 得到纯文本 | `StrOutputParser` | `str` |
| 得到约束结构 | `model.with_structured_output` | Pydantic 对象或字典，依 schema 而定 |
| 组合处理步骤 | `a \| b`、`RunnableLambda`、`RunnableParallel` | 末端组件的输出 |
| 让模型调用函数 | `@tool`、`create_agent` | Agent 状态字典 |
| 延续对话 | `checkpointer`、`thread_id` | 从检查点续接的状态 |
| 构建检索 | `Document`、切分器、Embeddings、VectorStore | 文档、向量或文档列表 |
| 找相关文本 | `store.as_retriever().invoke` | `list[Document]` |
| 观察中间过程 | `astream_events` 或 `agent.stream` | 事件字典或指定模式的数据 |

<a id="packages"></a>

## 1. 包、版本与安装

### 1.1 包名与导入名

| 安装包 | 本项目版本 | Python 导入前缀 | 职责 |
|---|---|---|---|
| `langchain` | 1.4.1 | `langchain` | Agent 工厂、中间件、模型初始化及常用类型重导出 |
| `langchain-core` | 1.6.3 | `langchain_core` | Runnable、消息、提示词、解析器、工具和检索等基础协议 |
| `langchain-openai` | 1.6.2 | `langchain_openai` | OpenAI 聊天模型和 Embeddings |
| `langchain-huggingface` | 1.2.2 | `langchain_huggingface` | Hugging Face 模型及 Embeddings |
| `langchain-qdrant` | 1.1.0 | `langchain_qdrant` | Qdrant 向量库集成 |
| `langchain-text-splitters` | 1.1.2 | `langchain_text_splitters` | 文本切分 |
| `langgraph` | 1.2.11 | `langgraph` | 图执行、Agent 底层运行、状态和恢复 |
| `langgraph-checkpoint` | 4.2.0 | `langgraph.checkpoint` | 检查点接口和内存实现 |
| `langchain-community` | 不作为本文环境前提 | `langchain_community` | 部分加载器和社区集成，需单独安装 |
| `langchain-classic` | 不作为本文环境前提 | `langchain_classic` | 旧版 Chains、旧 Memory 等兼容功能 |

这些包独立发版，版本号不应强行对齐。`langchain` 的 1.x 不意味着所有生态包必须是 1.x。v1 精简了主包命名空间，旧功能有一部分迁往 `langchain-classic`。[官方 v1 说明](https://docs.langchain.com/oss/python/releases/langchain-v1)

### 1.2 安装与版本检查

项目已有环境优先使用现有环境。以下是**供新建隔离环境参考的安装命令**，本文编写过程没有执行安装或升级：

```powershell
python -m pip install "langchain==1.4.1" "langchain-core==1.6.3" "langchain-openai==1.6.2" "langchain-text-splitters==1.1.2" "langchain-qdrant==1.1.0" "langgraph==1.2.11"
python -m pip check
python -c "import importlib.metadata as m; print(m.version('langchain')); print(m.version('langchain-core'))"
```

示例使用 Python 3.12；LangChain v1 要求 Python 3.10 或更高版本。需要 PDF 加载器时另装 `langchain-community` 和 `pypdf`，安装前检查其依赖是否会改变已有包。完整环境以项目 `environment.yml` 为准。

本文在线代码从环境变量读取 `OPENAI_API_KEY`、`OPENAI_CHAT_MODEL`，RAG 在线分支还需要 `OPENAI_EMBEDDING_MODEL`。后两个是**本文约定的变量名**，由示例代码读取，不是 LangChain 自动识别的配置。`OPENAI_BASE_URL` 为可选服务地址；兼容 OpenAI 协议不代表支持全部工具调用、结构化输出或流式特性。普通 Python 脚本不会自动读取 `.env`，若需读取应显式调用 `dotenv.load_dotenv()`。

### 1.3 怎样理解“公开接口”

本文使用公开导出路径和官方支持的公共方法；不调用 `_` 开头的内部实现。同名对象可以由多个模块重导出，例如消息对象可从 `langchain.messages` 或 `langchain_core.messages` 导入，不必混用。

官方网页持续更新，不能把其全部内容视为 1.0 就已有。本文以安装包实测签名为准，特别关注：`init_chat_model` 使用 `model_provider=`，`init_embeddings` 使用 `provider=`；流式事件版本与 Agent 图流版本是两套参数；`create_agent` 的高级参数会随次版本演进。文末列出本地核验差异。

来源：[API 总入口](https://reference.langchain.com/python)、[v1 迁移说明](https://docs.langchain.com/oss/python/migrate/langchain-v1)。

<a id="runnable"></a>

## 2. 统一调用接口与 Runnable

`Runnable[Input, Output]` 表示可执行组件。模型、提示词、解析器、Retriever 和组合链都实现了这一协议。**方法名字一致，不代表输入输出类型一致**：聊天模型吃消息，Retriever 吃查询字符串，提示词模板通常吃字典。

### 2.1 执行方法

以下公共方法继承自 `langchain_core.runnables.Runnable`；直接在组件实例上调用，无需另行导入函数。

| 常用调用形式 | 用途和重要参数 | 返回值 / 注意事项 |
|---|---|---|
| `r.invoke(input, config=None, **kwargs)` | 同步执行一次 | `Output` |
| `await r.ainvoke(input, config=None, **kwargs)` | 异步执行一次 | `Output`；默认实现可能在线程池运行同步方法 |
| `r.batch(inputs, config=None, return_exceptions=False)` | 多个独立输入；config 可为一个配置或逐项列表 | 与输入顺序一致的输出列表；不是供应商离线 Batch API |
| `await r.abatch(inputs, …)` | 异步批量 | 输出列表；并发不意味着服务端只收一个请求 |
| `r.batch_as_completed(inputs, …)` | 谁先完成先交付 | 迭代 `(输入下标, 输出或异常)` |
| `r.abatch_as_completed(inputs, …)` | 异步完成顺序交付 | 异步迭代器，用 `async for` |
| `r.stream(input, config=None)` | 增量输出 | 同步迭代器；默认实现可能只输出一次完整结果 |
| `r.astream(input, config=None)` | 异步增量输出 | 异步迭代器 |
| `r.astream_events(input, version="v2", …)` | 观察内部组件启动、流片段和结束 | 异步迭代事件字典；可按 `include_names/include_types/include_tags` 或相应 `exclude_*` 过滤 |
| `r.stream_events(input, version="v2", …)` | 本地 core 1.6.3 提供的同步事件流 | 同步迭代器；不能假定早期所有 1.x 版本都有 |

`return_exceptions=True` 时错误作为列表元素返回，调用方仍需检查 `isinstance(item, Exception)`。普通 `.stream()` 不保证每个片段恰好对应一个 token，也不保证整个链的所有步骤支持增量处理。

**局部示例，依赖 `r` 是已构建的 Runnable：**

```python
result = r.invoke(data, config={"tags": ["manual"], "max_concurrency": 4})
results = r.batch([data1, data2], return_exceptions=True)

async def watch(r, data):
    async for event in r.astream_events(data, version="v2"):
        print(event["event"], event["name"], event["data"])
```

v2 事件还含 `run_id`、`parent_ids`、`tags`、`metadata`。不要把 `event["data"]` 一律当文本，它会随事件类型变化。本地 core 1.6.3 还提供 v3 事件协议；本文示例显式用 v2，避免混用不同事件结构。

### 2.2 RunnableConfig 与配置绑定

导入：`from langchain_core.runnables import RunnableConfig`。它是配置 TypedDict，不是模型参数类。

| 配置键 | 作用 | 常见误解 |
|---|---|---|
| `tags: list[str]` | 追踪标签 | 不会加入提示词 |
| `metadata: dict` | 追踪附加信息 | 不等于 Agent state |
| `callbacks` | 回调处理器 | 不等于 token 返回值 |
| `run_name` | 本次运行名称 | 不更改工具或模型名称 |
| `max_concurrency` | 批量等支持该配置的操作的并发上限 | 不等于每秒请求上限，也不控制所有内部调度 |
| `recursion_limit` | 图等执行的步数限制 | 不等于输出 token 上限 |
| `configurable` | 运行时可配置字段；检查点的 `thread_id` 也放这里 | 任意放入的键不会自动修改模型参数 |

| 接口 | 用法、结果与边界 |
|---|---|
| `r.with_config(tags=[…], metadata={…})` | 返回绑定配置的新 Runnable |
| `r.bind(**kwargs)` | 绑定底层调用的关键字参数，返回 Runnable；例如支持 `stop` 的模型可 `.bind(stop=["END"])` |
| `r.with_retry(stop_after_attempt=3, retry_if_exception_type=(TimeoutError,))` | 返回失败后重试的包装器；3 表示总尝试次数，尽量包住最小失败步骤 |
| `r.with_fallbacks([backup], exceptions_to_handle=(TimeoutError,))` | 匹配异常时尝试备用 Runnable；备用输入输出须兼容 |
| `r.with_types(input_type=…, output_type=…)` | 声明类型信息，不能当成通用运行时验证器 |
| `r.with_listeners(on_start=…, on_end=…, on_error=…)` | 同步生命周期监听，回调接收运行记录；异步版本 `.with_alisteners` |
| `r.get_input_schema()` / `get_output_schema()` | 返回 Pydantic schema 类；可进一步 `.model_json_schema()` |
| `r.get_graph()` | 得到组合关系图对象，可用于查看执行结构 |

可配置模型/可序列化组件还可使用 `.configurable_fields`、`.configurable_alternatives`，不是每个任意 Runnable 子类都有这些方法。字段通过 `ConfigurableField` 声明后，才能用 `configurable` 选择或覆盖。用 `init_chat_model(..., configurable_fields=["temperature"])` 也可建立可配置模型，但具体模型必须支持该参数。

### 2.3 组合对象

导入前缀统一为 `langchain_core.runnables`。

| 对象或运算 | 常用构造 | 输入 → 输出 |
|---|---|---|
| `RunnableLambda` | `RunnableLambda(func, afunc=None)` | 函数输入 → 函数返回值；适合胶水代码 |
| `RunnableSequence` / `a \| b` | `prompt \| model \| parser` | 前一步输出成为后一步输入 |
| `RunnableParallel` | `RunnableParallel(left=a, right=b)` | 同一份输入 → `{"left": …, "right": …}` |
| `RunnablePassthrough` | `RunnablePassthrough()` | 原样返回输入 |
| `RunnablePassthrough.assign` | `.assign(extra=some_runnable)` | 输入字典 → 保留原字段并新增/覆盖字段的字典 |
| `RunnableBranch` | `RunnableBranch((condition, branch), default)` | 命中第一个条件就执行对应分支；最后一个是默认 Runnable |
| `.pick("key")` / `.pick(["a", "b"])` | 用在输出字典的 Runnable 后 | 取单字段值或字段子字典 |
| `.assign(key=…)` | 用在输出字典的 Runnable 后 | 对已有字典补充计算字段 |

**完整离线小例，可直接运行：**

```python
from langchain_core.runnables import (
    RunnableLambda, RunnableParallel, RunnablePassthrough, RunnableBranch,
)

double = RunnableLambda(lambda x: x * 2)
both = RunnableParallel(original=RunnablePassthrough(), doubled=double)
assert both.invoke(3) == {"original": 3, "doubled": 6}
enrich = RunnablePassthrough.assign(size=lambda x: len(x["text"]))
assert enrich.invoke({"text": "RAG"}) == {"text": "RAG", "size": 3}
route = RunnableBranch((lambda x: x < 0, lambda x: "negative"), lambda x: "nonnegative")
assert route.invoke(-1) == "negative"
assert double.batch([1, 2]) == [2, 4]
```

`RunnableParallel` 把输入广播给分支，不会把列表自动拆成多个任务；逐项执行可以用 `.map()`。普通字典只有在链的组合上下文中才可能被转换成并行 Runnable。

来源：[Runnable API](https://reference.langchain.com/python/langchain-core/runnables)、本地 `langchain_core/runnables/base.py` 及其公开方法签名。

<a id="models"></a>

## 3. 模型与消息

### 3.1 初始化与调用

| 导入路径 | 常用调用形式 | 返回值与说明 |
|---|---|---|
| `langchain.chat_models.init_chat_model` | `init_chat_model(model, model_provider=None, **kwargs)` | 聊天模型；开启配置字段时为可配置包装器。可用 `"openai:模型名"` 或显式 provider |
| `langchain_openai.ChatOpenAI` | `ChatOpenAI(model=…, api_key=…, base_url=…, timeout=…, max_retries=…)` | OpenAI 聊天模型；密钥通常交给环境变量 |
| `langchain_core.language_models.chat_models.BaseChatModel` | 子类实现模型协议 | 基类，用于扩展与类型标注，不直接作为可用模型实例 |

**局部在线示例：**

```python
import os
from langchain.chat_models import init_chat_model

model = init_chat_model(
    os.environ["OPENAI_CHAT_MODEL"], model_provider="openai", timeout=30,
)
message = model.invoke("用一句话解释 RAG。")
print(message.text)
```

`temperature`、`max_tokens`、工具并行等配置受具体提供商和模型约束，不能保证所有模型接受相同参数。模型 `.invoke` 可接收字符串、消息列表或 PromptValue；返回消息对象。`.stream` 返回消息片段，通常为 `AIMessageChunk`；拼接片段时可以相加，不能只拼 `content` 后丢掉工具调用信息。

### 3.2 消息类型与内容

以下类型统一可从 `langchain.messages` 导入；核心实现来自 `langchain-core`。

| 类型 / 属性 | 常用构造或访问 | 含义 |
|---|---|---|
| `SystemMessage` | `SystemMessage(content="回答简洁")` | 系统指令 |
| `HumanMessage` | `HumanMessage(content="问题")` | 用户消息；content 也可为多模态块列表 |
| `AIMessage` | `AIMessage(content="答案")` | 模型回复；可能只有工具调用而无文本 |
| `ToolMessage` | `ToolMessage(content="结果", tool_call_id="call_1")` | 工具执行结果，ID 必须对应请求 |
| `AIMessageChunk` | 由模型流式返回 | 可合并的消息增量，不等同完整消息 |
| `RemoveMessage` | `RemoveMessage(id="消息ID")` | 向支持消息 reducer 的状态发出删除指令，不是对模型说“删除” |
| `msg.content` | 字符串或块列表 | 原始消息内容，不保证是字符串 |
| `msg.text` | 字符串属性 | 提取文本部分；不要写成 `msg.text()` |
| `msg.content_blocks` | 标准化内容块列表 | 可含文本、工具调用等，具体能力依集成而定 |
| `msg.tool_calls` | 列表，元素常含 `name/args/id/type` | 解析好的工具调用；`args` 通常是字典 |
| `msg.invalid_tool_calls` | 调用解析失败的记录 | 不应当成有效参数直接执行 |
| `msg.usage_metadata` | 字典或 `None` | 常含 input/output/total token；供应商可能不提供 |
| `msg.response_metadata` | 字典 | 模型、结束原因等提供商信息，键不完全统一 |

**完整离线小例：**

```python
from langchain.messages import HumanMessage, AIMessage, ToolMessage

request = AIMessage(content="", tool_calls=[
    {"name": "add", "args": {"a": 2, "b": 3}, "id": "call_1", "type": "tool_call"}
])
reply = ToolMessage(content="5", tool_call_id="call_1")
history = [HumanMessage(content="2+3?"), request, reply]
assert history[-1].tool_call_id == history[-2].tool_calls[0]["id"]
```

### 3.3 绑定工具与结构化输出

| 方法 | 参数与返回值 | 容易误用的地方 |
|---|---|---|
| `model.bind_tools(tools, tool_choice=…, …)` | 工具列表 → 可调用的模型绑定对象，调用后仍返回消息 | 只让模型产生调用请求，不会在应用中执行 Python 函数 |
| `model.with_structured_output(schema, include_raw=False, …)` | Pydantic schema → Pydantic 实例；字典 schema 通常 → 字典 | 这是模型包装方法，不是事后解析任意文本 |
| 同上，`include_raw=True` | 返回含 `raw/parsed/parsing_error` 的字典 | 解析错误可在字段中；调用服务失败仍可能抛异常 |

`ChatOpenAI.with_structured_output` 还支持 `method`，例如 `json_schema`、`function_calling`、`json_mode`。具体 schema 限制、严格模式及默认值取决于安装版本与供应商；本文建议在业务中显式指定方法并确认模型支持。JSON mode 保证 JSON 形式的能力不等于完整 schema 验证。

**局部在线示例，前置：`model` 为支持原生 JSON schema 的 ChatOpenAI：**

```python
from pydantic import BaseModel, Field

class Answer(BaseModel):
    answer: str = Field(description="简短答案")
    supported: bool = Field(description="材料是否支持这个答案")

structured = model.with_structured_output(Answer, method="json_schema")
answer = structured.invoke("材料：RAG 先检索再生成。问题：是否包含检索？")
print(answer.answer, answer.supported)
```

### 3.4 消息处理函数

| 导入 | 常用调用 | 返回与注意事项 |
|---|---|---|
| `langchain_core.messages.convert_to_messages` | `convert_to_messages([("human", "你好")])` | 标准消息列表 |
| `langchain_core.messages.filter_messages` | `filter_messages(messages, include_types=["human"])` | 过滤后的消息列表；另有排除类型、名称、ID 参数 |
| `langchain_core.messages.merge_message_runs` | `merge_message_runs(messages)` | 合并可合并的相邻同类消息；工具消息有特殊处理 |
| `langchain_core.messages.trim_messages` | `trim_messages(messages, max_tokens=…, token_counter=…, strategy="last", …)` | 预算内的消息；可设 `include_system=True`、`start_on="human"` |
| `langchain_core.messages.utils.count_tokens_approximately` | `count_tokens_approximately(messages)` | 近似 token 数；不是供应商计费用量 |

裁剪后仍须保持 tool call 与 ToolMessage 配对；仅按字符截断可能生成无效对话。精确预算可采用模型支持的 token 计数方法，仍应预留输出空间。

来源：[模型指南](https://docs.langchain.com/oss/python/langchain/models)、[消息指南](https://docs.langchain.com/oss/python/langchain/messages)、[消息 API](https://reference.langchain.com/python/langchain-core/messages)。表中具体路径与签名按本地版本核对。

<a id="prompts"></a>

## 4. 提示词与输出解析

### 4.1 提示词

导入：`from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder`。

| 接口 | 输入与主要参数 | 返回 |
|---|---|---|
| `PromptTemplate.from_template("解释 {topic}")` | 字符串模板，默认 f-string 风格 | `PromptTemplate` |
| `prompt.format(topic="RAG")` | 模板变量 | `str` |
| `prompt.invoke({"topic": "RAG"})` | 输入字典 | `StringPromptValue`，可 `.to_string()` |
| `ChatPromptTemplate.from_messages([...])` | 角色与模板元组、消息、占位符 | `ChatPromptTemplate` |
| `chat.format_messages(**kwargs)` | 模板变量 | `list[BaseMessage]` |
| `chat.invoke(input_dict)` | 模板变量字典 | `ChatPromptValue`，可 `.to_messages()` |
| `MessagesPlaceholder("history", optional=True, n_messages=…)` | 对话消息占位 | 嵌入消息列表；限制消息数不等于 token 裁剪 |
| `prompt.partial(topic="RAG")` | 预填一部分变量，可用无参函数提供值 | 新模板；剩余变量调用时补齐 |

模板中的字面 JSON 花括号要转义为 `{{` 与 `}}`。`MessagesPlaceholder` 接收消息序列，不能直接塞入一段历史字符串。

**完整离线小例：**

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是 {subject} 助教。"),
    MessagesPlaceholder("history", optional=True),
    ("human", "{question}"),
]).partial(subject="RAG")
value = prompt.invoke({"question": "检索器输出什么？"})
assert len(value.to_messages()) == 2
print(value.to_string())
```

### 4.2 解析器

导入前缀：`langchain_core.output_parsers`。

| 对象 / 方法 | 用途与主要参数 | 返回 / 错误 |
|---|---|---|
| `StrOutputParser()` | 从模型消息提取文本 | `str`；适合链末端 |
| `JsonOutputParser()` | 解析 JSON 文本；可用 `.get_format_instructions()` 生成格式提示 | Python JSON 值，业务常用字典；不强制总是字典 |
| `PydanticOutputParser(pydantic_object=Schema)` | 解析并验证到 Pydantic 类型 | Schema 实例；验证失败通常包装为 `OutputParserException` |
| `parser.parse(text)` | 直接解析字符串 | 解析结果 |
| `parser.invoke(text_or_message)` | Runnable 风格调用 | 同类解析结果，可组合进 LCEL |
| `parser.get_format_instructions()` | 获取格式要求文字；适用于实现此方法的解析器 | `str`，需要主动放入提示词 |

解析器不会自动重新问模型，也不能证明模型内容真实。`JsonOutputParser` 的可选 Pydantic 配置主要用于 schema/格式说明，若需要严格 Pydantic 对象与验证，使用 `PydanticOutputParser`。

**完整离线小例：**

```python
from pydantic import BaseModel
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser

class Item(BaseModel):
    name: str
    count: int

assert JsonOutputParser().invoke('{"ok": true}') == {"ok": True}
parser = PydanticOutputParser(pydantic_object=Item)
item = parser.invoke('{"name": "chunk", "count": 2}')
assert item.count == 2
```

来源：[提示词 API](https://reference.langchain.com/python/langchain-core/prompts)、[解析器 API](https://reference.langchain.com/python/langchain-core/output_parsers)。

<a id="agents"></a>

## 5. 工具与 Agent

### 5.1 将函数暴露为工具

| 导入路径 | 常用接口 | 用途、参数与返回 |
|---|---|---|
| `langchain.tools.tool` | `@tool`、`@tool("name", args_schema=Schema)` | 函数 → BaseTool 子类实例；类型注解生成 schema，docstring 提供描述 |
| `langchain_core.tools.StructuredTool` | `.from_function(func=…, coroutine=…, name=…, description=…)` | 显式构建结构化工具；异步实现可通过 coroutine 提供 |
| `langchain.tools.BaseTool` | `.invoke(args)`、`await .ainvoke(args)` | 执行工具；输入可为参数字典或完整 ToolCall |
| `langchain.tools.ToolRuntime` | 工具签名中的 `runtime: ToolRuntime` | Agent 执行时注入运行信息，参数不会暴露给模型 |
| `langchain_core.tools.ToolException` | `raise ToolException("说明")` | 配合工具的 `handle_tool_error` 控制工具错误反馈 |

**完整离线小例：**

```python
from pydantic import BaseModel, Field
from langchain.tools import tool

class AddInput(BaseModel):
    a: int = Field(description="第一个整数")
    b: int = Field(description="第二个整数")

@tool(args_schema=AddInput)
def add(a: int, b: int) -> int:
    """将两个整数相加。"""
    return a + b

assert add.invoke({"a": 2, "b": 3}) == 5
message = add.invoke({
    "name": "add", "args": {"a": 2, "b": 3}, "id": "call_1", "type": "tool_call",
})
assert message.tool_call_id == "call_1"
print(add.name, add.description, add.args)
```

普通参数调用一般返回函数原始结果；携带完整 ToolCall 时会封装成 ToolMessage，以关联 `tool_call_id`。工具返回字符串不会自动更新 Agent 自定义状态；需要状态更新时可返回 `langgraph.types.Command(update=…)`，并按工具调用协议附带对应 ToolMessage。

`@tool(response_format="content_and_artifact")` 要求返回 `(面向模型的内容, 应用侧附加数据)`；完整 ToolCall 执行结果中的 `.artifact` 可保存文档列表等对象。工具 schema 与 Python 函数参数应匹配。`parse_docstring=True` 仅在文档字符串符合预期格式时使用，否则装饰阶段就可能报错。

`return_direct=True` 用于让支持这一语义的 Agent 在工具执行后结束，不等于工具调用本身返回纯文本。`handle_validation_error` 可处理参数验证问题；`handle_tool_error` 只针对约定的 ToolException，并非捕获所有程序错误。

### 5.2 create_agent

导入：`from langchain.agents import create_agent, AgentState`。

```text
create_agent(model, tools=None, *, system_prompt=None, middleware=(),
             response_format=None, state_schema=None, context_schema=None,
             checkpointer=None, store=None, …)
    → CompiledStateGraph
```

这是常用参数简写。返回的 Agent 是 LangGraph 编译图，支持调用、流式和状态接口，不是一个普通文本函数。

| 参数 | 传什么 | 作用 |
|---|---|---|
| `model` | 提供商模型字符串或 BaseChatModel 实例 | Agent 使用的聊天模型；工具 Agent 需要模型支持工具调用 |
| `tools` | BaseTool、带注解的函数或支持的工具字典列表 | 可用工具；通常先通过 `@tool` 定义更清晰 |
| `system_prompt` | 字符串或 SystemMessage | 固定系统指令；动态指令看中间件 |
| `middleware` | 中间件实例列表 | 修改模型/工具调用前后行为 |
| `response_format` | schema、ToolStrategy 或 ProviderStrategy | 规定最终结构化响应 |
| `state_schema` | 扩展 AgentState 的 TypedDict 类型 | 增加运行状态字段；需考虑并发更新规则 |
| `context_schema` | 通常为 dataclass 等上下文类型 | 声明每次调用传入的应用上下文 |
| `checkpointer` | 检查点实现 | 在同一 thread 下保存和恢复执行状态 |
| `store` | LangGraph Store | 跨会话数据存取，与消息检查点用途不同 |
| `interrupt_before/interrupt_after` | 节点名称列表 | 静态节点断点，常用于调试；工具审批优先使用对应中间件 |
| `name`、`debug` | 名称、布尔值 | 标识与调试；调试输出可能较多 |

**局部在线示例，前置：已有 `model` 和上节 `add`：**

```python
from langchain.agents import create_agent

agent = create_agent(model=model, tools=[add], system_prompt="算术问题使用 add 工具。")
state = agent.invoke({"messages": [{"role": "user", "content": "2 加 3？"}]})
print(state["messages"][-1].text)
```

Agent 输入常用 `{"messages": [...]}`，不是旧版 `{"input": "..."}`；结果常含消息列表。模型可能不调用工具，示例中的指令不能作为强制执行的程序保证。

### 5.3 Agent 结构化响应

导入：`from langchain.agents.structured_output import ToolStrategy, ProviderStrategy`。

| 接口 | 用法 | 适用与返回位置 |
|---|---|---|
| `ToolStrategy(Schema, handle_errors=True)` | 交给 `response_format` | 用工具调用实现结构化输出，模型须支持；结果在 `state["structured_response"]` |
| `ProviderStrategy(Schema, strict=…)` | 交给 `response_format` | 使用提供商原生结构化输出，要求模型支持 |
| `response_format=Schema` | 直接传 schema 类型 | 工厂按模型能力选择策略；能力识别和模型 profile 受版本影响 |

结构化结果不应从最后一条消息的 JSON 字符串自行猜测。业务必须固定策略时显式使用 Strategy；Pydantic schema 对应 Pydantic 对象，字典 schema 对应字典。schema 合法不代表答案有证据。[结构化输出指南](https://docs.langchain.com/oss/python/langchain/structured-output)

### 5.4 工具运行上下文

**局部示例：只定义工具，不进行网络请求。**

```python
from dataclasses import dataclass
from langchain.tools import tool, ToolRuntime

@dataclass
class Context:
    user_id: str

@tool
def current_user(runtime: ToolRuntime[Context]) -> str:
    """返回当前用户标识。"""
    return runtime.context.user_id

# 构建时：create_agent(model, [current_user], context_schema=Context)
# 调用时：agent.invoke({"messages": [...]}, context=Context(user_id="u1"))
```

`runtime.state` 是当前状态，`runtime.context` 是调用上下文，`runtime.store` 是可选数据存储，`runtime.stream_writer` 可写自定义进度。`runtime.config` 是执行配置。不要直接 `current_user.invoke({})` 并期待它凭空获得 Agent runtime。

来源：[工具指南](https://docs.langchain.com/oss/python/langchain/tools)、[Agent 指南](https://docs.langchain.com/oss/python/langchain/agents)、本地 `langchain/agents/factory.py` 与 `langchain_core/tools`。

<a id="memory"></a>

## 6. 中间件、状态与会话

### 6.1 常用内置中间件

以下均从 `langchain.agents.middleware` 导入，构造后放入 `create_agent(..., middleware=[...])`，构造本身不会执行模型。

| 类与常用构造 | 用途 | 边界 |
|---|---|---|
| `SummarizationMiddleware(model=…, trigger=("tokens", 4000), keep=("messages", 10))` | 达到阈值时压缩历史 | 摘要可能另调用模型；keep 是保留的最近部分，摘要不是无损备份 |
| `HumanInTheLoopMiddleware(interrupt_on={"tool_name": True})` | 在指定工具执行前请求人工决策 | 必须配置 checkpointer，并实现审批与 resume 流程 |
| `PIIMiddleware("email", strategy="redact", apply_to_input=True)` | 处理匹配的敏感信息 | 可选 block/redact/mask/hash；自定义类型需 detector，不保证识别所有敏感内容 |
| `ModelCallLimitMiddleware(run_limit=5, thread_limit=20)` | 限制一次运行或线程累计的模型调用 | 跨调用累计需保留状态；到达限制后的行为由 exit_behavior 控制 |
| `ToolCallLimitMiddleware(tool_name="search", run_limit=3)` | 限制工具调用 | 不指定名称则限制工具整体；不同退出策略行为不同 |
| `ModelRetryMiddleware(max_retries=2)` | 重试匹配的模型错误 | 这里是重试次数；与 Runnable 的总尝试次数含义不同 |
| `ToolRetryMiddleware(max_retries=2, tools=["search"])` | 重试指定工具 | 适合可安全重试的操作；写入类工具需考虑重复执行 |

底层 SDK、Runnable、Agent 中间件都可能重试，叠加后实际调用数会增加。摘要、PII、限流等中间件各有作用，不能用任意排列保证业务语义；应明确希望哪一步先处理请求。

### 6.2 自定义钩子

同样从 `langchain.agents.middleware` 导入装饰器或 `AgentMiddleware` 基类。

| 钩子 | 入参与返回 | 使用场景 |
|---|---|---|
| `before_agent` / `after_agent` | `(state, runtime) → 状态更新字典或 None` | 整次 Agent 运行前后处理 |
| `before_model` / `after_model` | `(state, runtime) → 状态更新字典或 None` | 每轮模型调用前后处理 |
| `wrap_model_call` | `(request, handler) → ModelResponse` | 包装调用、模型选择、请求修改；通常调用 `handler(request)` |
| `wrap_tool_call` | `(request, handler) → ToolMessage 或 Command` | 工具调用包装与错误反馈 |
| `dynamic_prompt` | `(request) → str 或 SystemMessage` | 按上下文生成系统提示词 |

类式扩展中异步钩子对应 `abefore_model`、`awrap_model_call` 等；使用异步 Agent 时确保相应实现可用，不要在异步函数内调用阻塞网络请求。包装钩子修改请求可用公开的 `request.override(...)`，不要直接依赖内部字段修改行为。

**局部定义示例：**

```python
from langchain.agents.middleware import dynamic_prompt, ModelRequest

@dynamic_prompt
def concise_prompt(request: ModelRequest) -> str:
    return "你是 RAG 助教。先回答结论，再说明证据。"

# 接入：create_agent(model, tools=[...], middleware=[concise_prompt])
```

### 6.3 检查点与状态查询

| 导入 / 对象 | 常用方法 | 返回与含义 |
|---|---|---|
| `langgraph.checkpoint.memory.InMemorySaver` | `InMemorySaver()` | 进程内检查点实现，进程退出即丢失 |
| Agent 编译图 | `agent.get_state(config)` / `await agent.aget_state(config)` | StateSnapshot，`.values` 为状态，另含下一步与任务信息 |
| Agent 编译图 | `agent.get_state_history(config)` / `.aget_state_history(config)` | 同步/异步历史快照迭代器 |
| Agent 编译图 | `agent.update_state(config, values, as_node=None)` | 写入状态更新并返回新检查点配置；遵循 reducer，不是任意整字典覆盖 |
| 同上 | `await agent.aupdate_state(...)` | 异步状态更新 |

**局部在线示例，前置：已有支持工具调用的 `model` 和 `add`：**

```python
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(model, [add], checkpointer=InMemorySaver())
config = {"configurable": {"thread_id": "manual-conversation-1"}}
agent.invoke({"messages": [("user", "记住我的名字是小林。") ]}, config=config)
result = agent.invoke({"messages": [("user", "我叫什么？") ]}, config=config)
print(result["messages"][-1].text)
snapshot = agent.get_state(config)
print(len(snapshot.values["messages"]))
```

同一会话复用同一 thread_id 与 checkpointer，后续只提交新增消息；重复提交全量历史可能重复追加。不同用户不能无条件共享 thread_id。长期保存要使用对应数据库的检查点实现并完成其初始化；仅添加 thread_id 不会自动获得磁盘持久化。

### 6.4 Store 与长期数据

`from langgraph.store.memory import InMemoryStore` 提供进程内键值存储。常见方法如下，异步版本在方法名前加 `a`：

| 方法 | 示例 | 返回 |
|---|---|---|
| `put(namespace, key, value)` | `store.put(("users", "u1"), "prefs", {"language": "zh"})` | 写入；通常为 None |
| `get(namespace, key)` | `store.get(("users", "u1"), "prefs")` | Item 或 None；数据在 `.value` |
| `search(namespace_prefix, query=…, filter=…, limit=…)` | 按命名空间和条件搜索 | SearchItem 列表；语义搜索需配置 embedding 索引 |
| `delete(namespace, key)` | 删除一项 | 通常为 None |

命名空间是字符串元组。Store 不会自动收集对话事实；需要工具或应用主动读写。InMemoryStore 同样不跨进程保存。

### 6.5 人工介入与恢复

**局部在线示例，前置：已有 `model` 和 `add`。此例审批计算工具，仅演示协议。**

```python
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

review_agent = create_agent(
    model, [add], checkpointer=InMemorySaver(),
    system_prompt="计算必须使用 add 工具。",
    middleware=[HumanInTheLoopMiddleware(interrupt_on={"add": True})],
)
config = {"configurable": {"thread_id": "review-demo"}}
paused = review_agent.invoke({"messages": [("user", "2+3？")]}, config=config)
if paused.get("__interrupt__"):
    for interrupt in paused["__interrupt__"]:
        print(interrupt.value)  # 展示待审批动作；此处不自动批准
    # 应用收集人的决定后，按待审批动作顺序构造 decisions。
    # 仅当恰有一个动作且人已批准时：
    # resumed = review_agent.invoke(
    #     Command(resume={"decisions": [{"type": "approve"}]}), config=config
    # )
```

允许的决定取决于 `interrupt_on` 配置，可含 approve、edit、reject。有多个动作时不能只提供一个决定；多个独立 interrupt 的恢复还需按 interrupt ID 处理。恢复必须使用相同 thread 配置，不要重新发起一条普通消息代替 resume。

### 6.6 Agent 流式输出的形状

以下约定针对 `agent.stream(..., version="v1")`，这是本地 LangGraph 1.2.11 的图流默认版本；它与 `astream_events(version="v2")` 的事件版本互不等价。

| `stream_mode` | 每项的内容 | 用途 |
|---|---|---|
| `"updates"` | 节点名 → 状态更新的字典 | 显示模型、工具执行步骤 |
| `"values"` | 完整状态值 | 观察每步状态 |
| `"messages"` | `(message_chunk, metadata)` | 观察模型消息流；metadata 可指明节点 |
| `"custom"` | 应用通过 stream_writer 写入的数据 | 工具进度 |
| `["updates", "messages"]` | `(mode, data)` | 同时消费多种模式；启用 subgraphs 后结构还会改变 |

**局部在线示例，前置：已建 `agent`，若启用检查点则传入对应 config：**

```python
for chunk, metadata in agent.stream(
    {"messages": [("user", "解释向量检索。")]},
    config={"configurable": {"thread_id": "stream-demo"}},
    stream_mode="messages", version="v1",
):
    if chunk.text:
        print(chunk.text, end="", flush=True)
```

本地也支持图流 `version="v2"`，使用统一的 `type/ns/data` 包装；不能沿用上面的二元组解包。Agent `.invoke(..., version="v2")` 的返回协议也不同；本文状态字典示例采用默认 v1。

来源：[中间件指南](https://docs.langchain.com/oss/python/langchain/middleware/overview)、[短期记忆](https://docs.langchain.com/oss/python/langchain/short-term-memory)、[流式指南](https://docs.langchain.com/oss/python/langchain/streaming)、[LangGraph 持久化](https://docs.langchain.com/oss/python/langgraph/persistence)。具体流形状对照本地 `langgraph/pregel/main.py`。

<a id="rag"></a>

## 7. RAG：加载、切分、向量化与检索

固定 RAG 的常见数据流是：`原文 → Document → 分块 Document → 向量库 → Retriever → 证据提示词 → 模型 → 文本`。这些环节可以单独运行和排错，不必先建立 Agent。

### 7.1 Document 与加载器

| 导入 / 接口 | 常用构造或调用 | 返回与说明 |
|---|---|---|
| `langchain_core.documents.Document` | `Document(page_content="正文", metadata={"source": "a.txt"}, id="d1")` | 文档对象；metadata 字段由应用约定，id 可选 |
| `langchain_core.document_loaders.BaseLoader` | 加载器基类 | 自定义加载器通常实现 lazy_load |
| Loader `.load()` / `await .aload()` | 读取全部内容 | `list[Document]`，大文件可能占用较多内存 |
| Loader `.lazy_load()` / `.alazy_load()` | 逐条加载 | 同步/异步迭代器；具体实现可能仍在内部一次读完整文件 |
| `langchain_community.document_loaders.TextLoader` | `TextLoader(path, encoding="utf-8")` | 文本加载器；额外需要 community 包 |
| `langchain_community.document_loaders.PyPDFLoader` | `PyPDFLoader(path, mode="page")` | PDF 加载器；额外需要 community、pypdf |

**可选集成片段，未纳入当前环境运行验收：**

```python
from langchain_community.document_loaders import PyPDFLoader

documents = PyPDFLoader("paper.pdf", mode="page").load()
for doc in documents[:2]:
    print(doc.metadata, doc.page_content[:80])
```

扫描 PDF 通常需要 OCR；安装 PDF 加载器不代表获得 OCR。页码字段及其起点由加载器定义，显示给用户前应核查。项目已有脚本也可直接使用 pypdf 读取后构建 Document，这不要求安装 community。

### 7.2 切分器

导入：`from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter, MarkdownHeaderTextSplitter`。

| 接口 | 参数 / 用途 | 返回与易错点 |
|---|---|---|
| `RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100, separators=…)` | 依次用段落、行、词等分隔符切分 | 切分器；默认长度函数是 len，因此通常按字符而非 token 计数 |
| `CharacterTextSplitter(separator="\n\n", …)` | 按指定分隔符切分合并 | 缺少分隔符时块可能超过目标大小 |
| `splitter.split_text(text)` | 输入字符串 | `list[str]` |
| `splitter.create_documents(texts, metadatas=None)` | 输入多段字符串及对应元数据 | `list[Document]` |
| `splitter.split_documents(documents)` | 切分并保留元数据 | `list[Document]`；原文档 id 不能直接当全部 chunk 的唯一 ID |
| `splitter.transform_documents(documents)` / `await .atransform_documents(documents)` | 文档转换协议 | 文档序列 |
| `RecursiveCharacterTextSplitter.from_tiktoken_encoder(encoding_name=…, chunk_size=…, …)` | 用 token 编码器度量块大小 | 需 tiktoken；编码器要匹配目标模型 |
| `MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "h1"), ("##", "h2")])` | 按 Markdown 标题切分 | `.split_text` 返回 Document 列表，与普通切分器不同 |

**完整离线小例：**

```python
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=40, chunk_overlap=8,
    separators=["\n\n", "\n", "。", " ", ""], add_start_index=True,
)
chunks = splitter.split_documents([
    Document(page_content="RAG先检索资料。" * 12, metadata={"source": "note.txt"})
])
assert len(chunks) > 1
assert all(c.metadata["source"] == "note.txt" for c in chunks)
```

`chunk_overlap` 应小于 chunk_size；实际重叠受分隔符和合并规则影响。`start_index` 是该输入文本中的字符位置，不是 PDF 页码或跨文件全局位置。

### 7.3 Embeddings

| 导入 / 接口 | 常用构造 | 说明 |
|---|---|---|
| `langchain.embeddings.init_embeddings` | `init_embeddings("openai:模型名")` 或 `init_embeddings(name, provider="openai")` | 返回对应 Embeddings 实例；注意这里参数叫 provider |
| `langchain_core.embeddings.Embeddings` | 子类协议 | 定义文档和查询向量化，通常不直接实例化 |
| `langchain_openai.OpenAIEmbeddings` | `OpenAIEmbeddings(model=…, dimensions=…)` | 远端向量化；dimensions 仅用于支持的模型 |
| `langchain_huggingface.HuggingFaceEmbeddings` | `HuggingFaceEmbeddings(model_name=…, model_kwargs={…}, encode_kwargs={…})` | 本地 sentence-transformers 模型；首次使用模型 ID 可能下载文件 |
| `langchain_core.embeddings.DeterministicFakeEmbedding` | `DeterministicFakeEmbedding(size=16)` | 按文本生成可重复测试向量，**没有语义能力** |

| 方法 | 输入 → 返回 | 说明 |
|---|---|---|
| `.embed_documents(texts)` | `list[str] → list[list[float]]` | 文档向量 |
| `.embed_query(text)` | `str → list[float]` | 查询向量；有些模型对查询和文档使用不同提示或编码方式 |
| `await .aembed_documents(texts)` | 同上 | 异步文档向量化 |
| `await .aembed_query(text)` | 同上 | 异步查询向量化 |

**局部集成片段，需要本地完整模型目录及 sentence-transformers：**

```python
import os
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name=os.environ["LOCAL_EMBEDDING_MODEL_PATH"],
    model_kwargs={"device": "cpu", "local_files_only": True},
    encode_kwargs={"normalize_embeddings": True},
)
vector = embeddings.embed_query("什么是检索增强生成？")
```

入库和查询需采用匹配的模型、维度及归一化约定；维度相同不代表两个模型的向量可互换。更换 embedding 模型通常需要重新建库。本例不在文档生成过程中下载权重或执行推理。

### 7.4 VectorStore 公共方法

基类：`langchain_core.vectorstores.VectorStore`；离线小规模示例可使用同模块的 `InMemoryVectorStore`。不同后端并不保证实现全部可选方法。

| 常用调用形式 | 输入与主要参数 | 返回 / 注意事项 |
|---|---|---|
| `StoreClass.from_documents(documents, embedding=emb, **backend_args)` | 文档与向量模型 | 已写入文档的实例；可能发出 embedding 请求 |
| `StoreClass.from_texts(texts, embedding=emb, metadatas=…, ids=…)` | 原始字符串 | 向量库实例 |
| `store.add_documents(documents, ids=…)` | 新增/写入文档 | ID 列表；同 ID 是覆盖还是冲突取决于后端 |
| `store.add_texts(texts, metadatas=…, ids=…)` | 直接写文本 | ID 列表 |
| `store.delete(ids=[…])` | 删除指定向量记录 | 后端可能返回 bool 或 None；不会自动删除原始文件 |
| `store.get_by_ids(ids)` | 按 ID 读取 | Document 列表；缺失 ID 可能跳过，不能依赖与输入顺序一致 |
| `store.similarity_search(query, k=4, **kwargs)` | 查询文本 | Document 列表，不带分数 |
| `store.similarity_search_with_score(query, k=4, **kwargs)` | 查询文本 | `(Document, float)` 列表，分数含义由后端/距离定义 |
| `store.similarity_search_with_relevance_scores(query, k=4, score_threshold=…)` | 相关性分数接口 | 约定为归一化相关性，需后端正确实现映射 |
| `store.similarity_search_by_vector(embedding, k=4)` | 已算好的查询向量 | Document 列表，省去本次查询向量化 |
| `store.max_marginal_relevance_search(query, k=4, fetch_k=20, lambda_mult=0.5)` | MMR 兼顾相关性和多样性 | Document 列表；lambda_mult 越接近 1 越重相关性 |
| `store.as_retriever(search_type="similarity", search_kwargs={"k": 4})` | 包装为 Retriever | VectorStoreRetriever |

常见异步对应包括 `afrom_documents`、`afrom_texts`、`aadd_documents`、`aadd_texts`、`adelete`、`aget_by_ids`、`asimilarity_search`、`asimilarity_search_with_score`、`asimilarity_search_with_relevance_scores`、`asimilarity_search_by_vector` 和 `amax_marginal_relevance_search`。其中一些是同步方法的异步包装，不能据此前缀推断后端具有原生异步连接。

不要统一假设原始 score 越大越相似：余弦相似度与欧氏距离方向不同；混合检索的融合分数也不等于相似概率。阈值应结合实际后端和数据校准。

### 7.5 Qdrant 集成

导入：`from langchain_qdrant import QdrantVectorStore, RetrievalMode`。Qdrant 客户端相关配置从 `qdrant_client` 导入，不属于 LangChain 参数类型。

| 接口 | 用途与关键参数 |
|---|---|
| `QdrantVectorStore.from_documents(docs, embedding=emb, location=":memory:", collection_name=…)` | 创建内存集合并写入；用于测试 |
| `QdrantVectorStore.from_documents(..., path="本地目录", …)` | 使用本地磁盘存储；目录重开后数据可保留 |
| `QdrantVectorStore.from_existing_collection(collection_name=…, embedding=emb, path=… 或 url=…)` | 连接已有集合，不重新写入 documents；向量名称、维度和距离要匹配 |
| `QdrantVectorStore(client=client, collection_name=…, embedding=emb)` | 包装已经建好的集合；此构造方式不会自动创建集合 |
| `retrieval_mode=RetrievalMode.DENSE/SPARSE/HYBRID` | 选择稠密、稀疏或混合检索；后两者需要 sparse_embedding 和匹配的集合向量配置 |
| 搜索的 `filter=models.Filter(...)` | 使用 Qdrant 原生过滤器，不是通用随意字典 |
| `.client` | 底层 QdrantClient；可访问其公开接口，结束本地示例后 `.close()` |

**完整离线小例，不调用模型服务、不建立磁盘集合：**

```python
from uuid import uuid4
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_qdrant import QdrantVectorStore
from qdrant_client import models

text = "RAG 先检索，再生成。"
doc_id = str(uuid4())
store = QdrantVectorStore.from_documents(
    [Document(page_content=text, metadata={"source": "manual"})],
    embedding=DeterministicFakeEmbedding(size=16),
    ids=[doc_id], location=":memory:", collection_name="manual_demo",
)
try:
    matches = store.similarity_search(
        text, k=1,
        filter=models.Filter(must=[models.FieldCondition(
            key="metadata.source", match=models.MatchValue(value="manual"),
        )]),
    )
    assert matches[0].page_content == text
    assert len(store.get_by_ids([doc_id])) == 1
    store.delete(ids=[doc_id])
    assert store.get_by_ids([doc_id]) == []
finally:
    store.client.close()
```

Qdrant point ID 通常使用 UUID 或无符号整数，示例使用 UUID 字符串，不使用任意 `"doc-1"`。默认 metadata payload 键为 `metadata`，因此过滤写 `metadata.source`。同一磁盘目录不要用多个本地客户端同时打开；并发服务部署使用服务端模式。此例只验证接口流程，不验证语义召回质量。

### 7.6 Retriever

导入：`from langchain_core.retrievers import BaseRetriever`。它是 `Runnable[str, list[Document]]`；使用 `.invoke(query)`、`await .ainvoke(query)` 和批量接口。

| 构造 | 效果 |
|---|---|
| `store.as_retriever(search_type="similarity", search_kwargs={"k": 4})` | 普通相似检索 |
| `store.as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 20, "lambda_mult": 0.5})` | 多样性检索，要求后端支持 MMR |
| `store.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": 4, "score_threshold": 0.7})` | 基于 relevance score 接口过滤，可能返回空列表 |

Retriever 的返回通常不带分数；需要展示分数应直接调用向量库的带分数方法，或自行定义输出结构。输入传字符串，不要把整份 `{"question": ...}` 字典直接交给标准 VectorStoreRetriever。

空检索结果应有明确业务分支；不要未经检查把空上下文交给模型再声称“基于资料”。metadata 是追溯依据，拼装提示词时应保留来源标识。

来源：[检索概念](https://docs.langchain.com/oss/python/langchain/retrieval)、[Qdrant 官方集成](https://docs.langchain.com/oss/python/integrations/vectorstores/qdrant)、[向量库 API](https://reference.langchain.com/python/langchain-core/vectorstores)、[切分器 API](https://reference.langchain.com/python/langchain-text-splitters)。可选加载器的构造参数已对照 [PyPDFLoader 官方源码](https://github.com/langchain-ai/langchain-community/blob/main/libs/community/langchain_community/document_loaders/pdf.py)；该源码链接指向持续更新的 main 分支，不代表本机已安装对应包。

<a id="operations"></a>

## 8. 缓存、回调、追踪与排错

### 8.1 模型缓存

导入：`from langchain_core.caches import InMemoryCache` 和 `from langchain_core.globals import set_llm_cache, get_llm_cache`。

| 接口 | 用途与返回 |
|---|---|
| `set_llm_cache(InMemoryCache())` | 设置进程级模型缓存；返回 None |
| `get_llm_cache()` | 获取当前缓存对象或 None |
| `set_llm_cache(None)` | 清除全局缓存设置 |
| `model` 构造参数 `cache=True/False/缓存实例` | 控制是否使用全局缓存或指定实现；默认行为看模型基类 |
| `cache.clear()` | 清理该缓存；缓存协议另有 lookup/update 和异步方法 |

模型缓存与检查点、向量库三者不同：缓存复用相同模型请求的结果，检查点恢复会话执行，向量库存储可检索资料。不要假设模型流式请求使用相同缓存机制。进程内缓存不跨进程保存；多用户应用应考虑输入与权限边界。

### 8.2 回调与 token 用量

| 导入 / 接口 | 使用方式 | 含义 |
|---|---|---|
| `langchain_core.callbacks.BaseCallbackHandler` | 子类实现 `on_chat_model_start`、`on_llm_new_token`、`on_llm_end`、`on_tool_start`、`on_tool_end`、`on_chain_error` 等 | 观察生命周期；处理器方法需要接受协议参数和 `**kwargs` |
| `langchain_core.callbacks.AsyncCallbackHandler` | 对应异步处理器 | 异步记录或转发事件 |
| `langchain_core.callbacks.UsageMetadataCallbackHandler` | 放入 `config={"callbacks": [handler]}` | 汇总支持标准用量元数据的调用；读 `.usage_metadata` |
| 模型返回的 `.usage_metadata` | 直接访问消息 | 单次调用用量；不自动换算费用 |

**局部在线示例，前置：已有 `model`：**

```python
from langchain_core.callbacks import UsageMetadataCallbackHandler

usage = UsageMetadataCallbackHandler()
model.invoke("一句话解释 RAG。", config={"callbacks": [usage], "tags": ["usage-demo"]})
print(usage.usage_metadata)
```

流式用量的获取取决于提供商配置，不能依赖每个片段都携带 token 数。对外输出实时文本时优先使用流式接口，回调通常用于观察和记录。

### 8.3 限流

导入：`from langchain_core.rate_limiters import InMemoryRateLimiter`。

```python
from langchain_core.rate_limiters import InMemoryRateLimiter

limiter = InMemoryRateLimiter(
    requests_per_second=2, check_every_n_seconds=0.1, max_bucket_size=2,
)
# 模型构造时传入：ChatOpenAI(model=..., rate_limiter=limiter)
```

返回限流器实例；底层可调用 `.acquire(blocking=True)` 或 `await .aacquire(blocking=True)` 获取许可，返回是否成功。它限制请求频率，不限制 token 数，也不是跨进程分布式限流器。`max_concurrency` 控制同时执行量，不能替代速率限制。

### 8.4 LangSmith 追踪

追踪属于 LangSmith 生态；可通过 `LANGSMITH_TRACING=true`、`LANGSMITH_API_KEY`、`LANGSMITH_PROJECT` 配置接入。开启后可能向配置的服务发送运行输入、输出和元数据。本文离线验收禁用追踪；手册不会更改项目的追踪设置。业务需要追踪时，根据实际数据范围配置脱敏和访问权限。[LangSmith 追踪说明](https://docs.langchain.com/langsmith/trace-with-langchain)

### 8.5 常见错误速查

| 现象 | 常见原因 | 检查方式 |
|---|---|---|
| `No module named langchain.chains` | 复制了旧版导入 | 查看第 10 章；使用 LCEL 或显式 classic 包 |
| `cannot import count_tokens_approximately` | 从消息包顶层导入 | 改从 `langchain_core.messages.utils` 导入 |
| 模板提示缺少变量 | 字典缺键、JSON 花括号未转义 | 查看 `prompt.input_variables` |
| `.content` 拼接时报类型错误 | 内容是块列表 | 用 `.text` 或按 `.content_blocks` 类型处理 |
| 模型输出了工具调用却没执行 | 只调用了 bind_tools | 手动执行调用循环，或使用 create_agent |
| ToolMessage 无法被提供商接受 | ID 不匹配、缺失请求消息、裁剪破坏配对 | 对照每个 tool_call 的 id |
| `structured_response` 不存在 | 未配置结构化响应、执行中断、未走到完成状态 | 检查 response_format 和 interrupt |
| 对话不记得上次内容 | 未设置 checkpointer 或更换 thread_id/实例 | 核对状态快照与配置 |
| 检查点报缺配置 | 配了 checkpointer 却没给 thread_id | 传 `configurable.thread_id` |
| 向量维度不匹配 | 更换模型/维度但沿用集合 | 对比 embedding 与集合定义，必要时重建 |
| 搜索结果为空 | 过滤器、阈值、字段路径或库为空 | 先去掉过滤和阈值检查基础召回 |
| Qdrant ID 报错 | 使用任意业务字符串作为 point ID | 用 UUID，并把业务 ID 放入 metadata |
| Qdrant 本地目录被占用 | 多个客户端同时打开 | 关闭客户端或改服务端部署 |
| 异步返回 coroutine 而非结果 | 忘记 await | `.ainvoke/.abatch` 使用 await；`.astream` 使用 async for |
| Notebook 报 asyncio.run 不能嵌套 | 当前已有事件循环 | Notebook 直接 await；普通脚本再 asyncio.run |
| 401、429、超时 | 密钥/配额/请求频率/网络问题 | 核对服务配置、限流与重试；不是提示词模板错误 |

来源：[缓存 API](https://reference.langchain.com/python/langchain-core/caches)、[回调 API](https://reference.langchain.com/python/langchain-core/callbacks)、[限流 API](https://reference.langchain.com/python/langchain-core/rate_limiters)。

<a id="examples"></a>

## 9. 可复制的完整示例

所有示例都是独立程序，不需要复制前面章节的变量。保存为任意 `.py` 文件后，在第 1 章对应环境中运行。9.1、9.3 默认离线；9.2 必须配置在线服务。在线分支只做静态与构造核验，未发起真实模型请求。

### 9.1 提示词 → 模型 → 文本，默认离线

依赖：`langchain-core`；开启在线模式还需要 `langchain-openai`。默认无环境变量要求；设 `LANGCHAIN_MANUAL_ONLINE=1` 时还需 `OPENAI_API_KEY` 和 `OPENAI_CHAT_MODEL`。

```python
import asyncio
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models.fake_chat_models import FakeListChatModel

online = os.getenv("LANGCHAIN_MANUAL_ONLINE") == "1"
if online:
    from langchain_openai import ChatOpenAI
    model = ChatOpenAI(model=os.environ["OPENAI_CHAT_MODEL"], timeout=30)
else:
    model = FakeListChatModel(responses=["RAG 先检索证据，再结合证据生成回答。"])

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是技术助教，请用一句话回答。"),
    ("human", "{question}"),
])
chain = prompt | model | StrOutputParser()
print(chain.invoke({"question": "什么是 RAG？"}))
print(chain.batch([{"question": "什么是检索？"}, {"question": "什么是生成？"}]))

async def main():
    answer = await chain.ainvoke({"question": "再解释一次 RAG。"})
    print(answer)
    async for part in chain.astream({"question": "什么是 RAG？"}):
        print(part, end="", flush=True)
    print()

if __name__ == "__main__":
    asyncio.run(main())
```

预期：离线模式输出固定回答，批量得到两个字符串，并演示异步与流式。FakeListChatModel 不分析问题，不能用它评估模型质量。在线模式会发出多次模型请求，并可能产生费用。

### 9.2 带工具、会话与结构化输出的 Agent

依赖：`langchain`、`langchain-openai`、`langgraph`、Pydantic。环境变量：`OPENAI_API_KEY`、`OPENAI_CHAT_MODEL`。模型需支持工具调用；本例采用 ToolStrategy。

```python
import os
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

@tool
def add(a: int, b: int) -> int:
    """计算两个整数的和。"""
    return a + b

class Calculation(BaseModel):
    result: int = Field(description="计算结果")
    explanation: str = Field(description="简短说明计算过程")

agent = create_agent(
    model=ChatOpenAI(model=os.environ["OPENAI_CHAT_MODEL"], timeout=30),
    tools=[add],
    system_prompt="遇到加法请使用 add 工具；最终给出结构化计算结果。",
    response_format=ToolStrategy(Calculation),
    checkpointer=InMemorySaver(),
)
config = {
    "configurable": {"thread_id": "calculation-demo"},
    "recursion_limit": 20,
}
state = agent.invoke({"messages": [("user", "计算 12 加 30。") ]}, config=config)
answer = state["structured_response"]
print(answer.result, answer.explanation)
for message in state["messages"]:
    if message.type == "tool":
        print("工具结果：", message.name, message.content)

next_state = agent.invoke(
    {"messages": [("user", "把刚才的结果再加 8。") ]}, config=config,
)
print(next_state["structured_response"].model_dump())
```

预期结果分别为 42 和 50，并能看到工具结果。真实模型可能调用失败、未遵从工具指令或触及递归限制，因此上线时要验证工具确实被调用及业务结果正确。InMemorySaver 仅在本进程中保留会话。

### 9.3 文档 → 切分 → 向量库 → 检索 → 生成，默认离线

依赖：`langchain-core`、`langchain-text-splitters`；在线分支另需 `langchain-openai`。默认无外部服务和模型文件，内存库不落盘。在线模式设 `LANGCHAIN_MANUAL_ONLINE=1`，并提供 `OPENAI_API_KEY`、`OPENAI_CHAT_MODEL`、`OPENAI_EMBEDDING_MODEL`。

```python
import os
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 仅供教学的关键词向量：它不是训练好的语义 embedding 模型。
class KeywordEmbeddings(Embeddings):
    vocabulary = ("检索", "生成", "向量", "缓存")

    def embed_query(self, text: str) -> list[float]:
        return [float(text.count(word)) for word in self.vocabulary] + [0.1]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

online = os.getenv("LANGCHAIN_MANUAL_ONLINE") == "1"
if online:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    embeddings = OpenAIEmbeddings(model=os.environ["OPENAI_EMBEDDING_MODEL"])
    model = ChatOpenAI(model=os.environ["OPENAI_CHAT_MODEL"], timeout=30)
else:
    embeddings = KeywordEmbeddings()
    model = FakeListChatModel(responses=[
        "演示回答：RAG 先检索资料，再依据证据生成回答。[rag-note]"
    ])

documents = [
    Document(page_content="RAG 先检索资料，再依据证据生成回答。", metadata={"source": "rag-note"}),
    Document(page_content="缓存用于复用相同请求的已有结果。", metadata={"source": "cache-note"}),
]
splitter = RecursiveCharacterTextSplitter(chunk_size=80, chunk_overlap=10)
chunks = splitter.split_documents(documents)
store = InMemoryVectorStore(embedding=embeddings)
ids = store.add_documents(chunks)
retriever = store.as_retriever(search_kwargs={"k": 1})

question = "RAG 如何检索和生成？"
hits = retriever.invoke(question)
if not hits:
    print("没有检索到资料，无法基于资料回答。")
else:
    context = "\n\n".join(
        f"[{doc.metadata['source']}] {doc.page_content}" for doc in hits
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", "仅根据以下资料回答；资料不足就说明不足。保留来源标识。\n{context}"),
        ("human", "{question}"),
    ])
    chain = prompt | model | StrOutputParser()
    print("检索来源：", [doc.metadata["source"] for doc in hits])
    print(chain.invoke({"context": context, "question": question}))
    if not online:
        assert hits[0].metadata["source"] == "rag-note"

store.delete(ids=ids)
assert store.get_by_ids(ids) == []
```

预期：默认模式检索到 `rag-note`，打印固定演示回答，随后删除记录。关键词向量只验证接口衔接，FakeListChatModel 也不会读取证据；切换为真实 embedding 与模型后才可开展语义检索与证据生成质量评估。正式 RAG 还需验证引用真实性、拒答、切分质量及权限过滤。

来源：[模型 Fake 实现 API](https://reference.langchain.com/python/langchain-core/language_models)、[内存向量库 API](https://reference.langchain.com/python/langchain-core/vectorstores)、本项目版本中对应公开类的离线执行结果。

<a id="migration"></a>

## 10. 旧版迁移与扩展索引

### 10.1 常见旧写法对应什么

| 旧教程常见内容 | 本文 1.x 写法 / 处理方式 | 是否简单改名 |
|---|---|---|
| `from langchain.chat_models import ChatOpenAI` | `from langchain_openai import ChatOpenAI` | 包归属变化，需要安装集成包 |
| `from langchain.embeddings import OpenAIEmbeddings` | `from langchain_openai import OpenAIEmbeddings` | 同上 |
| `langchain.text_splitter` | `langchain_text_splitters` | 拆包 |
| 模型 `.predict(...)`、直接 `model(...)` | `.invoke(...)`，异步 `.ainvoke(...)` | 返回常是消息，需取 `.text` |
| Retriever `.get_relevant_documents(...)` | `.invoke(query)` / `.ainvoke(query)` | 统一 Runnable 协议 |
| `LLMChain` | `prompt \| model \| parser` | LCEL 重组，输入输出需重新核对 |
| `RetrievalQA`、`ConversationalRetrievalChain` | 显式检索生成链，或检索工具 Agent | 不是直接替换名称；自行处理历史、证据与返回字段 |
| `initialize_agent`、旧 AgentExecutor 组合 | `create_agent` | 输入改为 messages，输出为状态；工具与记忆接入也不同 |
| `langgraph.prebuilt.create_react_agent` | `langchain.agents.create_agent` | 工厂和扩展点变化；旧 hooks 多转中间件 |
| `ConversationBufferMemory` 等旧 Memory | checkpointer + thread_id；必要时摘要中间件 | 状态生命周期不同，不能只换导入 |
| 必须维护旧 Chains | 安装并显式使用 `langchain_classic` | 兼容路线，不是新主包内同名接口仍存在 |

此表描述迁移方向，不承诺任意旧项目只替换 import 即可运行。`langchain-community` 也不是全部旧接口的替代品。[官方迁移指南](https://docs.langchain.com/oss/python/migrate/langchain-v1)

### 10.2 没有逐项展开的公开接口

以下提供方向和入口，按需求扩展，具体可用性查对应包版本：

| 模块 / 类型 | 值得继续查看的接口 | 官方入口 |
|---|---|---|
| Runnable 高级组合 | `RunnableGenerator`、`RunnableEach`、`RunnableSerializable`、`.as_tool()`、`.transform/.atransform`、schema JSON 方法 | [Runnable](https://reference.langchain.com/python/langchain-core/runnables) |
| Runnable 观察接口 | `.stream_events`、`.astream_log`；事件 v3 与相应对象协议 | [Runnable](https://reference.langchain.com/python/langchain-core/runnables) |
| 提示词扩展 | `FewShotPromptTemplate`、`FewShotChatMessagePromptTemplate`、example selectors | [Prompts](https://reference.langchain.com/python/langchain-core/prompts) |
| 输出解析扩展 | 列表、XML、工具调用解析器及流式解析 | [Output parsers](https://reference.langchain.com/python/langchain-core/output_parsers) |
| 消息序列化 | `message_to_dict`、`messages_to_dict`、`messages_from_dict`、消息块类型 | [Messages](https://reference.langchain.com/python/langchain-core/messages) |
| 序列化 | `langchain_core.load` 下 dump/load 相关接口；只处理可信且支持的对象 | [Core 总入口](https://reference.langchain.com/python/langchain-core) |
| 中间件扩展 | `ModelFallbackMiddleware`、`LLMToolSelectorMiddleware`、`TodoListMiddleware`、`ContextEditingMiddleware`、`ToolErrorMiddleware` | [Middleware API](https://reference.langchain.com/python/langchain/agents/middleware) |
| 中间件工具扩展 | Shell、文件搜索、ProviderToolSearch 等；这些能力不是通用 RAG 的必要依赖 | [Middleware API](https://reference.langchain.com/python/langchain/agents/middleware) |
| LangGraph | StateGraph、节点、边、interrupt、Command、检查点数据库实现 | [LangGraph](https://docs.langchain.com/oss/python/langgraph/overview) |
| 提供商集成 | OpenAI、Hugging Face、Anthropic、Ollama 等独立包 | [Integrations](https://reference.langchain.com/python/integrations/overview) |
| 其他向量库和 Retriever | 后端专有过滤、BM25、混合召回与重排 | [检索集成](https://docs.langchain.com/oss/python/integrations/retrievers) |

如果接口只在最新网页出现，先查看本机签名再决定采用：

```python
import inspect
from langchain.agents import create_agent
from langchain_core.runnables import Runnable

print(inspect.signature(create_agent))
print(inspect.signature(Runnable.astream_events))
print(inspect.getfile(create_agent))
```

<a id="verification"></a>

## 11. 核验记录与适用边界

### 11.1 版本证据

本次从项目 Conda 环境 `D:\Anaconda\envs\langchain\python.exe` 读取安装元数据，与项目声明的七个主要包版本一致。接口核验以该环境中的公开导出、`inspect.signature` 和安装源码为准；没有升级依赖。

官方文档与本地版本需要区分的具体事项：

- 在线参考页在核验时显示主包最新版本为 1.4.2，本项目是 1.4.1；本文不把网页上的全部新增内容视为本地已支持。
- 本地 `create_agent` 签名已包含 `cache`、`transformers` 等进阶参数，未在本文核心用法中展开；不能据此推断全部早期 1.x 都具备它们。
- core 1.6.3 的事件流支持 v1/v2/v3，本文使用 v2 事件字典；LangGraph 1.2.11 的图流支持 v1/v2，本文显式采用 v1 流示例。
- 本地 `ChatOpenAI.with_structured_output` 默认 method 为 `json_schema`；旧集成版本的默认值可能不同，本文示例显式指定。
- `count_tokens_approximately` 在本地从 `langchain_core.messages.utils` 导入；不依赖顶层重导出。
- `SummarizationMiddleware` 在本地同时接受 tuple 与 TriggerClause 等 trigger 写法；本文采用易读的 tuple 形式。

### 11.2 验证范围

| 检查项 | 本次结果 |
|---|---|
| 24 个 Python 代码块 | 全部通过 Python AST 语法检查；局部片段不当作独立程序执行 |
| 代码块中的 36 个不同生态导入 | 在本地环境导入成功；不包含未安装的可选 community 加载器 |
| 9 个完整离线示例 | Runnable、消息、提示词、解析器、工具、切分、Qdrant、9.1 调用链、9.3 RAG 全部执行成功 |
| 异步与消息处理 | ainvoke、abatch、astream、v2 astream_events、消息裁剪及 Store 读写通过 |
| Agent 协议补充验证 | 使用模拟聊天模型完成工具执行、检查点查询、人工审批中断和 Command 恢复 |
| 图流形状 | v1 的 updates/values/messages/多模式，以及 v2 的 type/ns/data 形状实测通过 |
| 中间件构造 | 摘要、PII、模型/工具调用限额、模型/工具重试构造通过；人工介入另有执行验证 |
| 在线接口构造 | 使用占位密钥构造 ChatOpenAI、bind_tools、结构化输出及 Agent，无真实服务请求 |
| 文档结构 | 目录锚点、代码围栏配对与相对链接检查通过 |
| 31 个不同官方外部链接 | HTTP 可达性检查通过；已替换失效的 PDF 加载器指南链接。可达性不等于网页内容永不变化 |

离线验证禁用 LangSmith 追踪，不读取项目密钥，也不运行在线分支。终端的中文显示受编码设置影响，文档文件以 UTF-8 保存。

在线模型响应质量、供应商配额和模型能力不属于离线检查可证明的内容；可选 community 加载器与本地 Hugging Face 模型推理未执行。在线示例需要读者使用自己的模型配置运行，并据真实返回结果验收。

本文未修改现有业务代码、环境文件或依赖配置。继续实践可对照 [RAG 知识学习手册](./RAG知识学习手册.md) 和 [RAG 系统构建方案](./RAG系统构建方案.md)。
