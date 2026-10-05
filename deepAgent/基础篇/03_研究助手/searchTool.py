import os

from deepagents import create_deep_agent
from dotenv import load_dotenv
from typing import Literal

from langchain.agents.middleware import TodoListMiddleware
from langchain_openai import ChatOpenAI
from langchain_protocol import ToolCall
from tavily import TavilyClient

load_dotenv()


TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
DEEPSEEK_API = os.getenv("DEEPSEEK_API")
DEEPSEEK_MODEL_NAME = os.getenv("DEEPSEEK_MODEL_NAME")

tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

# 定义搜索工具
def internet_search(
        query: str,
        max_results: int = 5,
        topic: Literal["general", "news", "finance"] = "general",
        include_raw_content: bool = False,
):
    """Run a web search for the given query.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return.
        topic: The topic category for the search.
        include_raw_content: Whether to include raw page content.
    """
    return tavily_client.search(
        # 注意形参，实参
        query,
        max_results=max_results,
        topic=topic,
        include_raw_content=include_raw_content,
    )


# 创建Agent并配置系统提示词
model = ChatOpenAI(
    model=DEEPSEEK_MODEL_NAME,
    api_key=DEEPSEEK_API,
    base_url="https://api.deepseek.com",
    # model_kwargs={
    #         "extra_body": {
    #             "thinking": {"type": "disabled"}   # 关闭思考模式（deepseek的思考模式与结构化输出不兼容）
    #         }
    #     }
)

research_instructions = """你是一位专业的研究员。
你的工作是进行深入研究，然后撰写一份完整的研究报告。

你可以使用 internet_search 工具搜索互联网获取信息。
"""

agent = create_deep_agent(
    model=model,
    tools=[internet_search],
    system_prompt=research_instructions,
    middleware=[TodoListMiddleware()],
    # “已启用TodoListMiddleware()”的复杂研究任务。没有传入 TodoListMiddleware 时，Agent 不会获得 write_todos 和 todos 状态；即使已经启用，模型也会根据任务决定是否实际调用工具，不能把图中的每一步当成固定执行协议。
)



# 运行Agent
result = agent.invoke(
    {"messages": [{"role": "user", "content": "什么是 LangGraph？"}]}
)

print(result["messages"][-1].content)




"""
总结：
    Middleware 让这个循环具备 Harness 能力：模型调用前可以整理上下文、注入技能索引或记忆内容，
    工具执行阶段可以检查权限、等待审批，或处理大体积结果。技能正文仍由模型按需通过 read_file 读取。
    不同 Middleware 使用不同 hooks
    
执行顺序：
    1.规划任务 — 因为示例显式启用了 TodoListMiddleware，Agent 可以调用 write_todos，把“研究 LangGraph”拆解为多个子步骤
    2.搜索信息 — 调用你提供的 internet_search 工具，执行多次网络搜索
    3.管理上下文 — 调用内置的 write_file 将大量搜索结果写入虚拟文件系统，避免上下文溢出
    4.委派子任务（如需要）— 调用内置的 task 工具，将复杂子任务委派给专门的子 Agent
    5.综合报告 — 从文件系统中读取整理好的信息，撰写最终报告
"""

