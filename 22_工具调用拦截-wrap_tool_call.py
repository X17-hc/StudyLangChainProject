from dotenv import load_dotenv
load_dotenv()


# @wrap_tool_call 让你在工具执行层面实现与 @wrap_model_call 类似的控制能力——重试、缓存、参数改写、结果后处理。

# 基本结构
# @wrap_tool_call 的结构与 @wrap_model_call 类似，接收 request 和 handler 两个参数：
from langchain.agents.middleware import wrap_tool_call

@wrap_tool_call
def my_tool_wrapper(request, handler):
    # request.tool_call: 包含工具名称和参数
    # request.tool: 工具对象本身
    # request.state: Agent 当前状态
    # request.runtime: 运行时上下文

    # 调用 handler(request) 才会真正执行工具
    result1 = handler(request)

    # result 是 ToolMessage 或 Command
    return result1




# 1.工具调用重试。工具执行可能因外部服务不稳定而失败，自动重试可以提升可靠性
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langchain.tools import tool


@wrap_tool_call
def retry_tool_on_error(request, handler):
    """工具调用失败时自动重试"""
    max_retries = 3
    last_result = None

    for attempt in range(max_retries):
        try:
            result = handler(request)
            # 检查是否是错误结果
            if hasattr(result, 'status') and result.status == "error":
                if attempt < max_retries - 1:
                    print(f"  [重试] 工具返回错误，第 {attempt + 1} 次重试...")
                    continue
            if attempt > 0:
                print(f"  [重试成功] 第 {attempt + 1} 次尝试")
            return result
        except Exception as e:
            if attempt < max_retries - 1:
                import time
                time.sleep((attempt + 1) * 2)
                print(f"  [重试] 异常 {e}，{attempt + 1} 次重试...")
            else:
                raise

    return last_result


# 模拟一个可能失败的工具
call_count = 0

@tool
def fetch_course_data(course_id: str) -> str:
    """从菜鸟教程 RUNOOB 获取课程数据。

    Args:
        course_id: 课程 ID
    """
    global call_count
    call_count += 1
    # 模拟前两次失败，第三次成功
    if call_count < 3:
        raise Exception(f"网络错误：无法连接到课程服务（第 {call_count} 次尝试）")
    return f"课程 {course_id}: Python3 基础教程，30 章，免费"


model = init_chat_model("deepseek:deepseek-v4-flash", temperature=0)
agent = create_agent(
    model=model,
    tools=[fetch_course_data],
    middleware=[retry_tool_on_error],
    system_prompt="你是菜鸟教程 RUNOOB 的课程助手。",
)

result = agent.invoke({
    "messages": [HumanMessage(content="帮我查一下课程 python-001 的信息")]
})
print(f"\n最终回复: {result['messages'][-1].content}")




# 2.修改工具参数。在工具执行之前动态修改参数，可以在不修改工具代码的情况下实现参数转换：
from langchain.agents.middleware import wrap_tool_call

@wrap_tool_call
def normalize_city_name(request, handler):
    """自动规范化城市名称（全角转半角、去除多余空格等）"""
    tool_call = request.tool_call

    # 只处理包含 city 参数的工具调用
    if "city" in tool_call.get("args", {}):
        city = tool_call["args"]["city"]
        # 规范化城市名：去空格、统一大小写
        normalized = city.strip().replace("　", "")  # 移除全角空格

        # 替换参数
        # 使用字典解包创建一个新的参数字典，其中 "city" 的值被替换为规范化后的 normalized。这样原参数里的其他字段保持不变。
        new_args = {**tool_call["args"], "city": normalized}
        # 同样通过解包原 tool_call 字典，并只替换 "args" 字段为新的 new_args，生成修改后的工具调用信息。
        new_tool_call = {**tool_call, "args": new_args}

        request = request.override(tool_call=new_tool_call)

    return handler(request)




# 3.工具结果缓存。对于重复的工具调用（相同工具 + 相同参数），可以缓存结果：
from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage

tool_cache = {}

@wrap_tool_call
def cache_tool_results(request, handler):
    """缓存工具执行结果"""
    # 生成缓存键：工具名 + 参数
    tool_name = request.tool_call.get("name", "unknown")
    tool_args = str(request.tool_call.get("args", {}))

    cache_key = f"{tool_name}:{tool_args}"

    # 检查缓存
    if cache_key in tool_cache:
        print(f"[工具缓存命中] {tool_name}")
        cached_content = tool_cache[cache_key]
        return ToolMessage(
            content=cached_content,
            tool_call_id=request.tool_call.get("id", ""),
            name=tool_name,
        )

    # 执行工具
    result = handler(request)

    # 存入缓存
    if hasattr(result, 'content'):
        tool_cache[cache_key] = result.content
        print(f"[工具缓存写入] {tool_name}，当前 {len(tool_cache)} 条")

    return result




# 4.工具调用日志与监控。记录所有工具调用的详细信息
import time
from langchain.agents.middleware import wrap_tool_call

@wrap_tool_call
def monitor_tool_performance(request, handler):
    """监控工具调用的性能指标"""
    tool_name = request.tool_call.get("name", "unknown")
    tool_args = request.tool_call.get("args", {})

    # 记录开始时间
    start_time = time.time()

    try:
        result = handler(request)
        elapsed = time.time() - start_time

        # 记录成功调用
        print(f"[监控] {tool_name}({tool_args}) 成功，耗时 {elapsed:.2f}s")
        return result
    except Exception as e:
        elapsed = time.time() - start_time
        # 记录失败调用
        print(f"[监控] {tool_name}({tool_args}) 失败，耗时 {elapsed:.2f}s，错误: {e}")




# 5.根据结果决定后续流程。
# 当 wrap_tool_call 返回 Command 时，可以通过 update 参数修改 Agent 状态。
#   使用 Command 可以直接向消息列表追加 AI 消息，然后 Agent 循环会自然结束。
from langchain.agents.middleware import wrap_tool_call
from langgraph.types import Command

@wrap_tool_call
def check_empty_result(request, handler):
    """如果工具返回空结果，直接结束 Agent，不浪费模型调用"""
    result = handler(request)

    # 检查是否返回了空结果
    if hasattr(result, 'content') and (
        "未找到" in str(result.content)
        or "无结果" in str(result.content)
        or "没有" in str(result.content)
    ):
        # 直接返回一个 Command 来更新状态
        # 添加一条 AI 消息说明情况
        from langchain.messages import AIMessage
        return Command(update={
            "messages": [
                AIMessage(content="抱歉，没有找到相关信息。请换个关键词试试。")
            ]
        })

    return result




# 总结：@wrap_model_call vs @wrap_tool_call
"""
    维度	        @wrap_model_call	                        @wrap_tool_call
    拦截目标	    模型调用	                                    工具执行
    request     内容	model、messages、tools、system_prompt	tool_call、tool、state、runtime
    返回类型	    ModelResponse 或 AIMessage	                ToolMessage 或 Command
    适用场景	    模型重试、降级、缓存、prompt 修改	            工具重试、缓存、参数改写、结果处理
"""