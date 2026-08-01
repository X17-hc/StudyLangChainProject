from unittest import result

from dotenv import load_dotenv
load_dotenv()


"""
langSmith 是什么。
当 Agent 在后台运行时，你看不到它内部发生了什么——调用了哪些模型、执行了哪些工具、每一步消耗了多少 Token。LangSmith 解决了这个"黑盒"问题。

    功能	        说明
    执行追踪	    记录 Agent 每一步的执行轨迹
    性能监控	    统计每次调用的耗时和 Token 消耗
    调试回放	    查看历史执行的详细信息
    评估测试	    创建测试集评估 Agent 表现
"""


# 快速开始
"""
    注册与安装： pip install langsmith
"""
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langchain.tools import tool

# 设置环境变量后，所有 Agent执行都会自动追踪，无需额外代码

@tool
def search_course(keyword: str) -> str:
    """ 搜索课程 """
    return f"搜索结果：{keyword} 相关课程"

model = init_chat_model("deepseek:deepseek-v4-flash", temperature=0)

agent = create_agent(
    model=model,
    tools=[search_course],
    system_prompt="你是菜鸟教程 RUNOOB 的助手",
)

# 这次执行会被自动记录到 LangSmith
result = agent.invoke({
    "messages": [HumanMessage(content="搜索 Python 课程")]
})

# APAC 控制台查看追踪：https://apac.smith.langchain.com
print("已完成，请到 APAC LangSmith 控制台查看追踪详情：")
print("https://apac.smith.langchain.com")


"""
在 LangSmith 控制台中，你可以看到每次 Agent 执行的完整轨迹：
    执行时间线：模型调用 → 工具调用 → 模型再调用的完整时间线
    输入/输出：每一步的输入消息和模型返回结果
    Token 用量：每次模型调用的 Token 消耗和费用估算
    延迟分析：每一步的耗时分布
    错误信息：如果某步出错，可以看到完整的错误堆栈
"""


# 手动创建追踪
from langsmith import traceable

# ###使用 @traceable 装饰器标记需要追踪的函数
@traceable
def process_user_query(query: str) -> dict:
    """处理用户查询（此函数会被单独追踪）"""
    # 预处理
    cleaned = query.strip().lower()
    # 调用 Agent
    result = agent.invoke({"messages": [HumanMessage(content=cleaned)]})
    return {
        "query": cleaned,
        "answer": result["messages"][-1].content,
    }

# 在 LangSmith 中会看到 process_user_query 作为一个独立步骤
result = process_user_query("  Python 课程推荐  ")
print(result["answer"])


"""
常用配置
    环境变量	                说明	                    示例
    LANGCHAIN_TRACING_V2	启用追踪（必须设为 true）	true
    LANGCHAIN_API_KEY	    LangSmith API Key	    lsv2_pt_xxx
    LANGCHAIN_PROJECT	    项目名称（用于分组追踪）	my-agent
    LANGCHAIN_ENDPOINT	    API 端点（默认即可）	    https://api.smith.langchain.com

生产环境建议将 LangSmith 的追踪采样率设低一些（避免记录所有请求造成成本过高），只在需要调试时开启完整追踪。
"""