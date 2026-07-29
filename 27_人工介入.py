from dotenv import load_dotenv
load_dotenv()



# 1.interrupt()——在工具中暂停执行. interrupt() 函数可以让工具执行到一半时暂停，等待外部输入后再继续：
from langgraph.types import interrupt

# 在工具中使用 interrupt() 暂停
def send_email(to: str, subject: str, body: str) -> str:
    """发送邮件（需要人工审批）"""
    # 暂停执行，向外部发送审批请求
    approval = interrupt({
        "action": "send_email",
        "to": to,
        "subject": subject,
        "body": body,
        "message": "请确认是否发送此邮件？"
    })

    # 等待外部传入 approval 后继续
    if approval.get("approved"):
        return f"邮件已发送给 {to}"
    else:
        return f"邮件发送已被拒绝：{approval.get('reason', '用户取消')}"

"""
interrupt() 的工作流程：
    1.工具调用 interrupt() → Agent 暂停执行
    2.外部系统获取中断信息，展示给用户
    3.用户做出决定后，通过 Command(resume=...) 恢复执行
    4.interrupt() 返回用户传入的值，工具继续执行
"""



# 2.完整示例 —— 审批流程
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage


@tool
def delete_course(course_name: str) -> str:
    """删除课程（需要审批）。

    Args:
        course_name: 要删除的课程名称
    """
    # 暂停并等待审批
    approval = interrupt({
        "action": "delete_course",
        "course": course_name,
        "message": f"确认删除课程《{course_name}》？此操作不可撤销。"
    })

    if approval.get("confirmed"):
        return f"课程《{course_name}》已删除"
    else:
        return f"删除操作已取消"


checkpointer = InMemorySaver()
model = init_chat_model("deepseek:deepseek-v4-flash", temperature=0)
agent = create_agent(
    model=model,
    tools=[delete_course],
    checkpointer=checkpointer,
    system_prompt="你是菜鸟教程 RUNOOB 的管理员助手。",
)

config = {"configurable": {"thread_id": "admin-001"}}

# 第一步：发起删除请求（会触发中断）
print("=== 开始执行 ===")
result = agent.invoke(
    {"messages": [HumanMessage(content="请删除课程《过时的 Java 教程》")]},
    config=config,
)

# 检查 Agent 是否暂停了
state = agent.get_state(config)
print(f"状态: {state.next}")  # ('tools',) 表示在 tools 节点暂停
print(f"中断信息: {state.tasks[0].interrupts}")

# 第二步：人工审批（模拟用户点击"确认"）
print("\n=== 人工审批 ===")
resume_value = {"confirmed": True, "operator": "管理员张三"}
result = agent.invoke(
    Command(resume=resume_value),
    config=config,
)
print(f"最终回复: {result['messages'][-1].content}")




# 3.interrupt_before / interrupt_after 参数
# 除了在工具中使用 interrupt()，你还可以在 create_agent() 中设置全局中断点：
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()
agent = create_agent(
    model="deepseek:deepseek-v4-flash",
    tools=[some_tool],
    checkpointer=checkpointer,

    # 在工具节点之前暂停（每次调用工具前都需要审批）
    interrupt_before=["tools"],

    # 在模型节点之后暂停（每次模型回复后都可以检查）
    # interrupt_after=["model"],
)

"""
    参数	                            暂停时机	        适用场景
    interrupt_before=["tools"]	    每次执行工具前	    所有工具调用都需要审批
    interrupt_before=["model"]	    每次模型调用前	    在模型处理前人工审查消息
    interrupt_after=["model"]	    每次模型回复后	    审查模型输出后再决定是否继续
    interrupt_after=["tools"]	    每次工具执行后	    检查工具结果后再决定下一步
"""



# 4.HITL的典型架构。在实际的 Web 应用中，HITL 通常这样实现：
# 后端：接收用户消息，处理到中断点，返回中断信息
def handle_user_message(thread_id: str, message: str):
    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke(
        {"messages": [HumanMessage(content=message)]},
        config=config,
    )
    state = agent.get_state(config)

    # 检查是否在等待审批
    if state.tasks and state.tasks[0].interrupts:
        return {
            "status": "pending_approval",
            "interrupt": state.tasks[0].interrupts[0].value,
            "thread_id": thread_id,
        }

    return {
        "status": "completed",
        "reply": result["messages"][-1].content,
    }


# 后端：处理用户审批
def handle_approval(thread_id: str, approved: bool, reason: str = ""):
    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke(
        Command(resume={"confirmed": approved, "reason": reason}),
        config=config,
    )
    return {"status": "completed", "reply": result["messages"][-1].content}

"""
    HITL 需要 Checkpointer 配合使用。因为 Agent 在 interrupt() 处暂停时，其状态必须被持久化，才能在恢复时正确继续执行。
"""