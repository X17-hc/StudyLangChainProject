from dotenv import load_dotenv
load_dotenv()

# 默认情况下，每次 agent.invoke() 都是独立的——Agent 不记得之前聊过什么。
#   Checkpointer（检查点保存器）让 Agent 能够记住对话历史，实现真正的多轮对话。



# 1.使用Checkpointer记住对话。添加 Checkpointer 后，同一 thread_id 下的对话会自动关联：
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage

# 创建一个内存 Checkpointer
checkpointer = InMemorySaver()

model = init_chat_model("deepseek:deepseek-v4-flash", temperature=0)
agent = create_agent(
    model=model,
    checkpointer=checkpointer,  # 传入 Checkpointer
    system_prompt="你是菜鸟教程 RUNOOB 的助手。",
)

# 使用 thread_id 来标识对话线程
"""
    thread_id 是关键。同一个 thread_id 下的对话是连续的，不同 thread_id 之间的对话完全隔离。
    这可以用一个 Agent 实例同时服务多个用户。
"""
config1 = {"configurable": {"thread_id": "user-001"}}
config2 = {"configurable": {"thread_id": "user-002"}}

# 第一轮
result1 = agent.invoke(
    {"messages": [HumanMessage(content="我叫小明，我在学 Python")]},
    config=config1,
)
print(f"第一轮: {result1['messages'][-1].content}")

# 第二轮——使用相同的 thread_id，Agent 记住了！
result2 = agent.invoke(
    {"messages": [HumanMessage(content="我叫什么名字？我在学什么？")]},
    config=config2,
)
print(f"第二轮: {result2['messages'][-1].content}")



# 2.Checkpointer的工作原理
"""
    Checkpointer 在每次 Agent 执行后自动保存状态快照（checkpoint）。下一次使用相同 thread_id 调用时，自动从最近的 checkpoint 恢复状态。
    具体工作流程：
        1.调用 agent.invoke()，传入 config（含 thread_id）
        2.Agent 检查是否有该 thread_id 的 checkpoint
        3.如果有，加载历史消息，追加新消息后继续
        4.执行完成后，自动保存新的 checkpoint
"""
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage

checkpointer = InMemorySaver()
agent = create_agent(
    model=init_chat_model("deepseek:deepseek-v4-flash", temperature=0),
    checkpointer=checkpointer,
    system_prompt="你是菜鸟教程 RUNOOB 的助手。",
)

config = {"configurable": {"thread_id": "demo-001"}}

# 模拟多轮对话
questions = [
    "我叫小明",
    "我在学 Python",
    "帮我总结一下关于我的信息",
]

for i, q in enumerate(questions, 1):
    result = agent.invoke(
        {"messages": [HumanMessage(content=q)]},
        config=config,
    )

    # 查看 checkpoint 状态
    state = agent.get_state(config)
    print(f"\n第 {i} 轮后:")
    print(f"  消息数: {len(state.values.get('messages', []))}")
    print(f"  下一步: {state.next}")
    print(f"  回复: {result['messages'][-1].content[:80]}...")



# 3.Checkpointer 类型
"""
    类型	                存储位置	        持久化	        适用场景
    InMemorySaver	    内存	            否（重启丢失）	    开发调试、测试
    SqliteSaver	        SQLite 文件	    是	            单机部署、小规模应用
    PostgresSaver	    PostgreSQL	    是	            生产环境、多实例共享
"""

# 3.1 SqliteSaver 示例
from langgraph.checkpoint.sqlite import SqliteSaver

# SQLite 持久化——重启后数据不丢失
checkpointer = SqliteSaver.from_conn_string("conversations.db")

agent = create_agent(
    model="deepseek:deepseek-v4-flash",
    checkpointer=checkpointer,
)

# 用法和 InMemorySaver 完全相同
config = {"configurable": {"thread_id": "user-001"}}
result = agent.invoke(
    {"messages": [{"role": "user", "content": "你好"}]},
    config=config,
)
# 重启程序后，相同 thread_id 的对话依然存在


# 4 管理对话线程
# 4.1 查看对话状态
# 查看对话状态
state = agent.get_state(config)
print(f"下一步: {state.next}")  # () 表示空闲
print(f"消息数: {len(state.values.get('messages', []))}")

# 查看对话历史
for msg in state.values.get("messages", []):
    print(f"  [{msg.type}] {str(msg.content)[:60]}")


# 4.2 创建新线程
# 不同的 thread_id = 不同的独立对话
config_alice = {"configurable": {"thread_id": "alice"}}
config_bob = {"configurable": {"thread_id": "bob"}}

# Alice 的对话
agent.invoke(
    {"messages": [HumanMessage(content="我是 Alice")]},
    config=config_alice,
)

# Bob 的对话——完全独立，不知道 Alice 说了什么
agent.invoke(
    {"messages": [HumanMessage(content="我是 Bob")]},
    config=config_bob,
)

# 验证隔离性
alice_state = agent.get_state(config_alice)
bob_state = agent.get_state(config_bob)
print(f"Alice 对话消息数: {len(alice_state.values['messages'])}")
print(f"Bob 对话消息数: {len(bob_state.values['messages'])}")




# 5.更新状态 —— 手动修改对话。有时你需要手动修改对话状态，比如清空对话、插入系统消息等
"""
    update_state() 的参数会通过 add_messages reducer 处理（对 messages 字段而言），所以新消息会追加而不是覆盖。如果需要覆盖整个状态，需要使用不同的方法。
"""
from langchain.messages import SystemMessage

# 更新状态——插入一条系统消息
agent.update_state(
    config,
    {
        "messages": [
            SystemMessage(content="（用户升级到了 VIP 会员）")
        ]
    }
)
# 之后的对话会包含这条插入的消息
