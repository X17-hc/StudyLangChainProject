from dotenv import load_dotenv
load_dotenv()


"""
需求分析
    功能	        实现方式
    知识库问答	RAG 检索 + 模型回答
    订单查询	    @tool 工具函数
    对话记忆	    SqliteSaver Checkpointer
    敏感内容过滤	@before_model Middleware
    人工转接	    HITL interrupt()
"""
# 完整代码
# 文件路径：customer_service_bot.py
# pip install langchain langchain-deepseek langchain-chroma chromadb
from dotenv import load_dotenv
load_dotenv()

from typing import Annotated
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import before_model, after_model, wrap_tool_call
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import interrupt, Command


# ========== 1. 准备知识库 ==========

knowledge_base = [
    "菜鸟教程 RUNOOB 创立于 2013 年，是国内领先的免费编程学习平台。",
    "平台提供 300+ 套教程，涵盖 Python、Java、HTML、CSS、JavaScript 等。",
    "Python3 基础教程共 30 章，累计学习人次超 500 万。课程完全免费。",
    "VIP 会员费用为 ¥99/月，¥799/年，包含视频课程和一对一答疑服务。",
    "退款政策：购买 7 天内且在 3 节课以内可全额退款。",
    "平台支持在线编程环境，无需安装任何软件即可编写运行代码。",
    "客服工作时间：周一至周五 9:00-18:00，周末 10:00-16:00。",
]

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
chunks = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30
                                         ).create_documents(knowledge_base)
vector_store = Chroma.from_documents(chunks, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})


# ========== 2. 定义工具 ==========

@tool
def search_kb(query: str) -> str:
    """搜索菜鸟教程知识库，获取关于平台、课程、政策等官方信息。

    Args:
        query: 搜索问题或关键词
    """
    docs = retriever.invoke(query)
    if not docs:
        return "未找到相关信息，建议转接人工客服。"
    return "\n".join(f"- {doc.page_content}" for doc in docs)


# 模拟订单数据库
orders_db = {
    "ORD-2024-001": {"user": "小明", "item": "VIP 年费会员",
                      "amount": 799, "status": "已完成", "date": "2024-01-15"},
    "ORD-2024-002": {"user": "小明", "item": "Python 实战课程",
                      "amount": 199, "status": "配送中", "date": "2024-03-20"},
}


@tool
def query_order(order_id: str) -> str:
    """根据订单号查询订单状态和详情。

    Args:
        order_id: 订单号，如 ORD-2024-001
    """
    order = orders_db.get(order_id.upper())
    if not order:
        return f"未找到订单 {order_id}。请确认订单号是否正确。"
    return (f"订单 {order_id}：{order['item']} | "
            f"金额 ¥{order['amount']} | "
            f"状态 {order['status']} | "
            f"日期 {order['date']}")


@tool
def transfer_to_human(reason: str) -> str:
    """将用户转接给人工客服。

    Args:
        reason: 转接原因
    """
    approval = interrupt({
        "action": "transfer_to_human",
        "reason": reason,
        "message": f"用户请求转接人工客服，原因：{reason}。是否转接？"
    })
    if approval.get("confirmed"):
        return (f"已为您转接人工客服，预计等待 {approval.get('wait_time', 3)} 分钟。"
                f"工单号：TK-{approval.get('ticket_id', 'N/A')}")
    return "转接已取消，我继续为您服务。"


# ========== 3. 定义 Middleware ==========

@before_model
def content_guard(state, runtime):
    """过滤用户输入中的不当内容"""
    last_msg = state["messages"][-1] if state.get("messages") else None
    if not last_msg:
        return None
    content = str(getattr(last_msg, 'content', ''))
    blocked = ["黄X", "X博", "违法"]
    for word in blocked:
        if word in content:
            return {
                "jump_to": "end",
                "messages": [HumanMessage(content="抱歉，我不能处理这个请求。")]
            }
    return None


@after_model
def auto_signature(state, runtime):
    """自动追加客服签名"""
    msgs = state.get("messages", [])
    if not msgs:
        return None
    last = msgs[-1]
    if last.type == "ai" and last.content and not (
        hasattr(last, 'tool_calls') and last.tool_calls
    ):
        from langchain.messages import AIMessage
        return {"messages": [AIMessage(
            content=last.content
            + "\n\n---\n菜鸟教程 RUNOOB 客服中心 | 工作时间 9:00-18:00"
        )]}
    return None


# ========== 4. 创建 Agent ==========

checkpointer = SqliteSaver.from_conn_string("customer_service.db")
model = init_chat_model("deepseek:deepseek-v4-flash", temperature=0)

agent = create_agent(
    model=model,
    tools=[search_kb, query_order, transfer_to_human],
    middleware=[content_guard, auto_signature],
    checkpointer=checkpointer,
    system_prompt="""你是菜鸟教程 RUNOOB 的智能客服"小菜"。

## 你的职责
1. 热情接待每一位用户，用"您"称呼
2. 关于平台信息、课程内容、政策等问题，使用 search_kb 查询
3. 关于订单查询，使用 query_order 工具
4. 遇到无法解决的问题，使用 transfer_to_human 转接人工

## 行为准则
- 回答简洁，每次 2-3 句话
- 不知道的就查询知识库，查不到就诚实告知
- 保持友好亲切的语气""",
)


# ========== 5. 对话接口 ==========

def chat(thread_id: str, message: str) -> str:
    """处理用户消息并返回回复"""
    config = {"configurable": {"thread_id": thread_id}}

    # 运行 Agent
    result = agent.invoke(
        {"messages": [HumanMessage(content=message)]},
        config=config,
    )

    # 检查是否需要转接（HITL）
    state = agent.get_state(config)
    if state.tasks and state.tasks[0].interrupts:
        interrupt_info = state.tasks[0].interrupts[0].value
        return f"[需要审批] {interrupt_info.get('message', '')}"

    return result["messages"][-1].content


# ========== 6. 测试 ==========

if __name__ == "__main__":
    user_id = "user_xiaoming"

    print("=== 测试 1：知识库查询 ===")
    print(chat(user_id, "Python3 教程有多少章？"))
    print()

    print("=== 测试 2：订单查询 ===")
    print(chat(user_id, "我的订单 ORD-2024-001 状态是什么？"))
    print()

    print("=== 测试 3：VIP 咨询 ===")
    print(chat(user_id, "VIP 会员多少钱？"))
    print()

    print("=== 测试 4：测试记忆 ===")
    print(chat(user_id, "我刚才问过什么问题？"))



"""
运行结果：
    === 测试 1：知识库查询 ===
    根据知识库信息，Python3 基础教程共 30 章，累计学习人次超过 500 万，课程完全免费。
    ---
    菜鸟教程 RUNOOB 客服中心 | 工作时间 9:00-18:00
    
    === 测试 2：订单查询 ===
    您的订单 ORD-2024-001：VIP 年费会员，金额 ¥799，状态 已完成，日期 2024-01-15。
    ---
    菜鸟教程 RUNOOB 客服中心 | 工作时间 9:00-18:00
    
    === 测试 3：VIP 咨询 ===
    VIP 会员费用为 ¥99/月或 ¥799/年，包含视频课程和一对一答疑服务。
    ---
    菜鸟教程 RUNOOB 客服中心 | 工作时间 9:00-18:00
    
    === 测试 4：测试记忆 ===
    您刚才咨询了 Python3 教程章节数、订单 ORD-2024-001 的状态，以及 VIP 会员的价格。
    还有什么需要帮您的吗？
    ---
    菜鸟教程 RUNOOB 客服中心 | 工作时间 9:00-18:00
"""



"""
    特性	            在项目中的使用
    RAG 检索	        search_kb 工具 + Chroma 向量存储
    工具调用	        query_order、transfer_to_human
    Checkpointer	SqliteSaver 持久化对话，实现多轮记忆
    Middleware	    before_model 内容过滤 + after_model 签名追加
    HITL	        interrupt() 实现人工转接审批
"""