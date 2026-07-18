from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()


"""
@wrap_model_call 是中间件（Middleware）中最强大的钩子。

@wrap_model_call 不像 before/after 那样只是观察，而是可以完全控制模型的执行过程——重试、降级、缓存、甚至跳过模型直接用预设回复。
"""

# 1.理解handler 回调
# @wrap_model_call 的核心是一个 handler 回调函数。调用 handler(request) 才会真正执行模型；不调用则跳过模型。

# wrap_model_call 的基本结构
# request: 包含模型、消息、工具等全部信息
# handler: 一个可调用对象，执行它才会真正调用模型
from langchain.agents.middleware import wrap_model_call

@wrap_model_call
def my_middleware(request, handler):
    # 在模型调用前可以做任何事
    print("模型即将被调用...")

    # 调用 handler(request) 才真正执行模型
    response = handler(request)

    # 在模型调用后可以做任何事
    print("模型调用完成")

    return response



# 2.重试机制
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage

@wrap_model_call
def retry_on_error(request, handler):
    """模型调用失败时自动重试，最多 3 次"""
    max_retries = 3
    last_error = None

    for attempt in range(max_retries):
        try:
            result = handler(request)
            if attempt > 0:
                print(f"  [重试成功] 第 {attempt + 1} 次尝试")
            return result
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2  # 递增等待：2s, 4s
                print(f"  [重试] 第 {attempt + 1} 次失败，{wait_time}秒后重试...")
                import time
                time.sleep(wait_time)

    # 所有重试都失败了
    raise last_error


model = init_chat_model("deepseek:deepseek-v4-flash", temperature=0)
agent = create_agent(
    model=model,
    middleware=[retry_on_error],
    system_prompt="你是菜鸟教程 RUNOOB 的助手。",
)

result = agent.invoke({
    "messages": [HumanMessage(content="介绍一下菜鸟教程")]
})
print(f"\n回复: {result['messages'][-1].content[:100]}...")




# 3.模型降级/故障转移
from langchain.agents.middleware import wrap_model_call
from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage


# 主模型和备用模型
primary_model = init_chat_model("deepseek:deepseek-v4-flash", temperature=0)
fallback_model = init_chat_model("deepseek:deepseek-v4-flash", temperature=0,
                                  max_tokens=100)  # 降级模型限制 Token


@wrap_model_call
def fallback_on_error(request, handler):
    """主模型失败时自动切换到备用模型"""
    try:
        # 尝试用主模型
        return handler(request)
    except Exception as e:
        print(f"[降级] 主模型失败: {e}，切换到备用模型...")

        # 覆盖 request 中的 model，切换到备用模型
        # request.override() 是一个不可变方法——它返回一个新的 request 副本，不会修改原始对象。这确保了每次调用都是独立和安全的。
        request = request.override(model=fallback_model)
        try:
            return handler(request)
        except Exception as e2:
            print(f"[降级] 备用模型也失败了: {e2}")
            # 两个模型都失败，返回友好提示
            return AIMessage(
                content="抱歉，服务暂时不可用，请稍后重试。"
            )




# 4.缓存模型响应
from langchain.agents.middleware import wrap_model_call
from langchain.messages import AIMessage
from langchain.agents import *

# 简单的内存缓存
cache = {}

@wrap_model_call
def cache_responses(request, handler):
    """缓存模型响应，相同问题不重复调用"""
    # 使用最后一条用户消息的内容作为缓存键
    messages = request.messages
    if not messages:
        return handler(request)

    # 生成缓存键
    last_content = str(messages[-1].content) if hasattr(messages[-1], 'content') else ""
    cache_key = last_content[:200]  # 截断超长内容

    # 检查缓存
    if cache_key in cache:
        print(f"[缓存命中] 直接返回缓存结果")
        cached = cache[cache_key]
        return AIMessage(content=f"{cached}\n\n*(来自缓存)*")

    # 缓存未命中，调用模型
    result1 = handler(request)

    content = None
    if hasattr(result1, 'content'):
        content = result1.content
    elif hasattr(result1, 'model_response'):
        # ExtendedModelResponse 内通常有 .model_response，它是一个 AIMessage
        inner = result1.model_response
        if hasattr(inner, 'content'):
            content = inner.content

    if content:
        cache[cache_key] = content
        print(f"[缓存未命中] 已存入缓存，当前 {len(cache)} 条")

    return result1


model = init_chat_model("deepseek:deepseek-v4-flash", temperature=0)
agent = create_agent(
    model=model,
    middleware=[cache_responses],
    system_prompt="你是菜鸟教程 RUNOOB 的助手，回答要简洁。",
)

# 第一次查询（缓存未命中）
result = agent.invoke({
    "messages": [{"role": "user", "content": "Python 是什么？"}]
})
print(f"第一次: {result['messages'][-1].content[:80]}...\n")

# 第二次查询相同问题（缓存命中）
result = agent.invoke({
    "messages": [{"role": "user", "content": "Python 是什么？"}]
})
print(f"第二次: {result['messages'][-1].content[:80]}...")




# 5.修改 request——动态注入系统消息
from datetime import datetime
from langchain.agents.middleware import wrap_model_call
from langchain.messages import SystemMessage

@wrap_model_call
def inject_time_context(request, handler):
    """在每次模型调用前注入当前时间信息"""
    now = datetime.now()
    time_context = (
        f"当前时间：{now.strftime('%Y年%m月%d日 %H:%M')}。"
        f"今天是{['周一','周二','周三','周四','周五','周六','周日'][now.weekday()]}。"
    )

    # 在原有 system_message 基础上追加时间信息
    if request.system_message:
        new_content = f"{request.system_message.content}\n\n{time_context}"
    else:
        new_content = time_context

    # 用 override 创建新的 request
    new_request = request.override(
        system_message=SystemMessage(content=new_content)
    )

    return handler(new_request)




# 6.多个 wrap_model_call 的组合.
# 多个 wrap_model_call 中间件会自动按顺序组合——第一个在最外层：
from langchain.agents.middleware import wrap_model_call

@wrap_model_call
def outer_middleware(request, handler):
    """最外层中间件"""
    print("[外层] 开始")
    result = handler(request)       # 这里会进入 inner_middleware
    print("[外层] 结束")
    return result

@wrap_model_call
def inner_middleware(request, handler):
    """内层中间件"""
    print("  [内层] 开始")
    result = handler(request)       # 这里才真正调用模型
    print("  [内层] 结束")
    return result

"""
    执行顺序：
    [外层] 开始
      [内层] 开始
        → 真正调用模型
      [内层] 结束
    [外层] 结束

    多个 wrap_model_call 就像洋葱一样层层包裹。最外层最先执行、最后返回。
    这让你可以组合多个独立的功能——比如外层做缓存检查，内层做重试，互不干扰。
"""

