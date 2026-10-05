from dotenv import load_dotenv
load_dotenv()

"""
    方式	    用户体验	适用场景
    invoke()	等待 → 一次性看到完整结果	脚本、API、批处理
    stream()	实时看到每一个 Token	聊天界面、实时展示
"""

# 1.stream_mode="messages"——逐 Token 流式
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage

model = init_chat_model("deepseek:deepseek-v4-flash")
agent = create_agent(
    model=model,
    system_prompt="你是菜鸟教程 RUNOOB 的助手。",
)

# stream_mode="messages" 逐个 Token 返回
print("实时流式输出：")
for msg_chunk, metadata in agent.stream(
    {"messages": [HumanMessage(content="用一句话介绍菜鸟教程 RUNOOB")]},
    stream_mode="messages",
):
    # msg_chunk 是 AIMessageChunk
    # 每个 chunk 只包含一小段内容
    if msg_chunk.content:
        print(msg_chunk.content, end="", flush=True)
print()  # 最后换行


# 2.理解metadata
print("查看 metadata 信息：\n")
for msg_chunk, metadata in agent.stream(
    {"messages": [HumanMessage(content="你好，介绍一下你自己")]},
    stream_mode="messages",
):
    if msg_chunk.content and len(msg_chunk.content) > 5:
        print(f"内容: {msg_chunk.content}")
        print(f"来源节点: {metadata.get('langgraph_node')}")
        print(f"消息类型: {type(msg_chunk).__name__}")
        break  # 只看第一个有意义的 chunk



# 2.stream_mode="updates"——逐步查看 Agent 执行过程
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage


@tool
def search_course(keyword: str) -> str:
    """在菜鸟教程 RUNOOB 搜索课程"""
    courses = {
        "python": "Python3 基础教程（30章，20小时）",
        "html": "HTML 基础教程（25章，15小时）",
    }
    return courses.get(keyword.lower(), "未找到相关课程")


agent = create_agent(
    model=init_chat_model("deepseek:deepseek-v4-flash", temperature=0),
    tools=[search_course],
    system_prompt="你是菜鸟教程 RUNOOB 的课程顾问。",
)

# 使用 updates 模式查看每一步
print("=== Agent 执行过程 ===\n")
for chunk in agent.stream(
    {"messages": [HumanMessage(content="帮我查一下 Python 课程")]},
    stream_mode="updates",
):
    for node_name, update in chunk.items():
        print(f"[{node_name}]", end=" ")
        if update and "messages" in update:
            for msg in update["messages"]:
                if msg.type == "ai":
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        calls = [tc['name'] for tc in msg.tool_calls]
                        print(f"请求调用: {calls}")
                    elif msg.content:
                        print(f"回复: {msg.content[:80]}")
                elif msg.type == "tool":
                    print(f"工具返回 [{msg.name}]: {msg.content}")




# 3.stream_mode="custom"——发送自定义事件
from langchain.agents.middleware import before_model, after_model

@before_model
def notify_before(state, runtime):
    """在模型调用前发送自定义事件"""
    runtime.stream_writer({
        "type": "status",
        "message": "正在思考...",
    })
    return None


@after_model
def notify_after(state, runtime):
    """在模型调用后发送自定义事件"""
    last_msg = state["messages"][-1] if state.get("messages") else None
    has_tools = hasattr(last_msg, 'tool_calls') and last_msg.tool_calls

    if has_tools:
        tool_names = [tc['name'] for tc in last_msg.tool_calls]
        runtime.stream_writer({
            "type": "status",
            "message": f"正在调用工具: {', '.join(tool_names)}...",
        })
    else:
        runtime.stream_writer({
            "type": "status",
            "message": "回答已完成",
        })
    return None


agent = create_agent(
    model=init_chat_model("deepseek:deepseek-v4-flash", temperature=0),
    tools=[search_course],
    middleware=[notify_before, notify_after],
    system_prompt="你是菜鸟教程 RUNOOB 的课程顾问。",
)

# 使用 stream_mode=["updates", "custom"] 同时接收两种事件
print("=== 混合流式输出 ===\n")
for mode, chunk in agent.stream(
    {"messages": [HumanMessage(content="查一下 Python 课程")]},
    stream_mode=["updates", "custom"],
):
    if mode == "custom":
        print(f"[自定义事件] 状态: {chunk['message']}")
    elif mode == "updates":
        for node_name, update in chunk.items():
            if update and "messages" in update:
                for msg in update["messages"]:
                    if msg.type == "ai" and msg.content:
                        print(f"[回复] {msg.content}")



# 4.异步流式输出。在 Web 服务中，使用异步流式可以避免阻塞事件循环：
import asyncio

async def stream_agent():
    """异步流式运行 Agent"""
    agent = create_agent(
        model=init_chat_model("deepseek:deepseek-v4-flash"),
        system_prompt="你是菜鸟教程 RUNOOB 的助手。",
    )

    full_response = ""
    async for msg_chunk, metadata in agent.astream(
        {"messages": [HumanMessage(content="一句话介绍菜鸟教程")]},
        stream_mode="messages",
    ):
        if msg_chunk.content:
            full_response += msg_chunk.content
            print(msg_chunk.content, end="", flush=True)

    print(f"\n\n完整回复长度: {len(full_response)} 字")


# 运行异步函数
asyncio.run(stream_agent())



# FastAPI集成示例
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage

app = FastAPI()
agent = create_agent(
    model=init_chat_model("deepseek:deepseek-v4-flash"),
    system_prompt="你是菜鸟教程 RUNOOB 的助手。",
)


@app.get("/chat")
async def chat(message: str):
    """聊天接口，返回 SSE 流式响应"""
    async def generate():
        async for msg_chunk, metadata in agent.astream(
            {"messages": [HumanMessage(content=message)]},
            stream_mode="messages",
        ):
            if msg_chunk.content:
                # SSE 格式：data: xxx\n\n
                yield f"data: {msg_chunk.content}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
    )

# 启动：uvicorn main:app --reload



"""
stream_mode 速查表
    模式	        粒度	            迭代对象	                    典型用途
    messages	Token 级	    (AIMessageChunk, metadata)	打字效果、实时聊天
    updates	    节点级	        {node_name: state_update}	展示思考过程
    values	    节点级（全量）	    完整 state	                状态快照、调试
    custom	    自定义	        任意 dict	                进度通知、状态推送
    debug	    详细	            调试信息	                    开发阶段排查问题
"""
