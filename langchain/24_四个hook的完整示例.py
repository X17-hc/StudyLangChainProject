from dotenv import load_dotenv
load_dotenv()

from langchain.agents.middleware import (
    before_agent, after_agent, before_model, after_model
)
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langchain.tools import tool


# ----- 定义所有钩子 -----

@before_agent
def init_session(state, runtime):
    """开始：初始化会话"""
    print(">>> 会话开始")
    return None


@before_model
def pre_model_check(state, runtime):
    """每次模型调用前"""
    msg_count = len(state.get("messages", []))
    print(f"  [model前] 消息数: {msg_count}")
    return None


@after_model
def post_model_check(state, runtime):
    """每次模型调用后"""
    last = state["messages"][-1] if state.get("messages") else None
    if last and hasattr(last, 'tool_calls') and last.tool_calls:
        print(f"  [model后] 需要工具调用")
    return None


@after_agent
def finish_session(state, runtime):
    """结束：清理资源"""
    total = len(state.get("messages", []))
    print(f"<<< 会话结束，共 {total} 条消息")
    return None


# ----- 创建 Agent -----

@tool
def get_weather(city: str) -> str:
    """查询天气"""
    return f"{city}: 晴"


model = init_chat_model("deepseek:deepseek-v4-flash", temperature=0)
agent = create_agent(
    model=model,
    tools=[get_weather],
    middleware=[init_session, pre_model_check, post_model_check, finish_session],
    system_prompt="你是助手。",
)

result = agent.invoke({
    "messages": [HumanMessage(content="杭州天气？")]
})
print(f"\n最终回复: {result['messages'][-1].content}")



"""
Middleware 钩子总结
    钩子	                执行次数	    何时使用	                    关键能力
    before_agent	    1 次	    权限检查、输入预处理、资源初始化	可 jump_to="end" 提前终止
    before_model	    每次循环	    消息裁剪、内容过滤、上下文注入	可 jump_to 控制流程
    wrap_model_call	    每次循环	    重试、降级、缓存、prompt 修改	完全控制模型执行
    after_model	        每次循环	    响应审核、内容追加、日志	    可替换模型输出
    wrap_tool_call	    每次工具调用	工具重试、缓存、参数改写	    完全控制工具执行
    after_agent	        1 次	    输出格式化、统计分析、清理	    最终状态修改
"""