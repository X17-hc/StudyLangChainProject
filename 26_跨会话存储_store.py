from dotenv import load_dotenv
load_dotenv()


# Checkpointer 解决了"单个对话内记忆"的问题。但如果你需要在不同对话之间共享数据——比如用户偏好、学习进度——就需要用到 Store。
"""
Checkpointer vs Store
    维度	        Checkpointer	            Store
    作用域	    单个对话线程（thread_id）	    跨所有对话线程
    数据类型	    Agent 状态快照（自动管理）	    任意键值数据（手动管理）
    典型用途	    多轮对话记忆	                用户偏好、知识库、配置
    数据组织	    thread_id → checkpoint	    (namespace, key) → value
"""


# 1.Store 的基本操作。Store 使用 命名空间 + 键 的层级结构来组织数据：
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()

# 写入数据：put(namespace, key, value)
# namespace 是元组，key 是字符串，value 是字典
store.put(
    ("users", "user_001"),           # 命名空间
    "preferences",                    # 键
    {                                 # 值
        "theme": "dark",
        "language": "zh-CN",
        "level": "入门",
    }
)

store.put(
    ("users", "user_001"),
    "progress",
    {
        "completed_courses": ["HTML 基础", "Python 基础"],
        "total_hours": 35,
    }
)

# 读取数据：get(namespace, key)
prefs = store.get(("users", "user_001"), "preferences")
print(f"偏好设置: {prefs.value}")

progress = store.get(("users", "user_001"), "progress")
print(f"学习进度: {progress.value}")

# 搜索数据：search(namespace)
all_user_data = store.search(("users", "user_001"))
print(f"\n用户的所有数据 ({len(all_user_data)} 项):")
for item in all_user_data:
    print(f"  {item.key}: {item.value}")

# 删除数据：delete(namespace, key)
store.delete(("users", "user_001"), "preferences")
deleted = store.get(("users", "user_001"), "preferences")
print(f"\n删除后: {deleted}")  # None



# 2.在Agent中使用store。将Store传给create_agent()，Agent中的所有工具都能通过InjectedStore访问它
from typing import Annotated
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langchain.tools import tool, InjectedStore
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage

# 创建 Store 并预置数据
store = InMemoryStore()
store.put(("runoob", "courses"), "catalog", {
    "Python3 基础教程": {"price": "免费", "hours": 20, "level": "入门"},
    "Python 数据分析": {"price": "会员", "hours": 30, "level": "进阶"},
    "Java 面向对象": {"price": "免费", "hours": 25, "level": "进阶"},
})

store.put(("runoob", "users"), "user_vip_001", {
    "name": "小明",
    "membership": "VIP",
    "joined": "2024-01-15",
})


@tool
def query_course_info(
    course_name: str,
    store: Annotated[BaseStore, InjectedStore()],
) -> str:
    """查询菜鸟教程 RUNOOB 中课程的详细信息。

    Args:
        course_name: 课程名称
    """
    item = store.get(("runoob", "courses"), "catalog")
    catalog = item.value if item else {}

    if course_name in catalog:
        info = catalog[course_name]
        return (
            f"《{course_name}》- 价格：{info['price']}，"
            f"时长：{info['hours']}小时，难度：{info['level']}"
        )
    return f"未找到课程《{course_name}》"


@tool
def get_user_membership(
    user_id: str,
    store: Annotated[BaseStore, InjectedStore()],
) -> str:
    """查询用户会员信息。

    Args:
        user_id: 用户 ID
    """
    item = store.get(("runoob", "users"), user_id)
    if item is None:
        return f"未找到用户 {user_id}"

    user = item.value
    return (
        f"用户 {user['name']}，{user['membership']} 会员，"
        f"注册日期 {user['joined']}"
    )


model = init_chat_model("deepseek:deepseek-v4-flash", temperature=0)
agent = create_agent(
    model=model,
    tools=[query_course_info, get_user_membership],
    store=store,
    system_prompt="你是菜鸟教程 RUNOOB 的课程顾问。",
)

# 查询课程信息（数据来自 Store）
result = agent.invoke({
    "messages": [HumanMessage(content="Python3 基础教程多少钱？")]
})
print(f"查询课程: {result['messages'][-1].content}")

# 查询用户信息（数据来自 Store）
result = agent.invoke({
    "messages": [HumanMessage(content="帮我查一下用户 user_vip_001 的信息")]
})
print(f"查询用户: {result['messages'][-1].content}")




# 3.Store的持久化
# 开发阶段
from langgraph.store.memory import InMemoryStore
store = InMemoryStore()

# 生产环境（需要 PostgreSQL）
# from langgraph.store.postgres import PostgresStore
# store = PostgresStore.from_conn_string("postgresql://...")



"""
Store 使用建议
    场景	        namespace 示例	            key 示例	        说明
    用户偏好	    ("users", user_id)	        preferences	    主题、语言、通知设置
    学习进度	    ("users", user_id)	        progress	    已完成课程、学习时长
    知识库	    ("kb", collection)	        doc_id	        文档、FAQ、产品信息
    会话摘要	    ("sessions", thread_id)	    summary	        长对话的摘要，供 Checkpointer 之外的场景使用
"""