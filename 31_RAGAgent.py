from dotenv import load_dotenv
load_dotenv()


# 创建 Retriever 工具
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ----- 步骤 1：准备知识库 -----

# 模拟菜鸟教程 RUNOOB 的知识文档
knowledge_docs = [
    "菜鸟教程（RUNOOB）创立于 2013 年，是一个完全免费的编程学习平台。",
    "平台已上线 300+ 套教程，涵盖前端、后端、数据库、移动开发等领域。",
    "Python3 基础教程是平台最受欢迎的课程，累计学习人次超过 500 万。",
    "Python3 基础教程共 30 章，包含环境搭建、基本语法、函数、类、异常处理等内容。",
    "HTML 基础教程共 25 章，从 HTML 基本结构讲到表单与多媒体元素。",
    "菜鸟教程支持在线运行代码，学习者无需安装任何软件即可编写和运行代码。",
    "平台提供移动端适配，用户可以在手机上随时随地学习编程。",
    "菜鸟教程的会员服务提供视频课程、项目实战、一对一答疑等增值服务。",
]

# 切分
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200, chunk_overlap=30
)
chunks = text_splitter.create_documents(knowledge_docs)

# 向量化存储
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# ----- 步骤 2：创建检索工具 -----

@tool
def search_knowledge_base(query: str) -> str:
    """在菜鸟教程 RUNOOB 知识库中搜索相关信息。

    当用户询问关于菜鸟教程的具体信息时（如课程数量、平台历史、功能特性等），
    必须使用此工具查询知识库获取准确信息。

    Args:
        query: 搜索关键词或问题
    """
    docs = retriever.invoke(query)
    if not docs:
        return "知识库中未找到相关信息。"

    results = []
    for i, doc in enumerate(docs, 1):
        results.append(f"[{i}] {doc.page_content}")

    return "\n\n".join(results)


# ----- 步骤 3：创建 RAG Agent -----

model = init_chat_model("deepseek:deepseek-v4-flash", temperature=0)
agent = create_agent(
    model=model,
    tools=[search_knowledge_base],
    system_prompt="""你是菜鸟教程 RUNOOB 的智能客服助手。

## 规则
1. 当用户询问关于菜鸟教程的具体信息时，必须使用 search_knowledge_base 工具查询
2. 基于检索到的信息回答，不要编造知识库中没有的内容
3. 如果知识库中没有相关信息，诚实地告诉用户
4. 回答要友好、简洁、准确""",
)

# ----- 步骤 4：测试 -----

questions = [
    "菜鸟教程是什么时候创立的？",
    "Python3 基础教程有多少章？",
    "菜鸟教程一共有多少套教程？",
]

for q in questions:
    result = agent.invoke({"messages": [HumanMessage(content=q)]})
    print(f"Q: {q}")
    print(f"A: {result['messages'][-1].content}")
    print("-" * 60)