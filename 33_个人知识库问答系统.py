from dotenv import load_dotenv

load_dotenv()

"""
系统设计
    文档加载：支持 Markdown、TXT、PDF 多种格式
    向量检索：Chroma 持久化存储，支持增量更新
    引用来源：回答中附带来源文档和片段位置
    流式输出：逐 Token 显示回答
"""

import os
from pathlib import Path
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader


def _build_embeddings() -> OpenAIEmbeddings:
    """构建 Embedding 客户端。

    优先使用 OPENAI_API_KEY；若未配置则回退到智谱 OpenAI 兼容接口（ZHIPU_API_KEY）。
    """

    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        return OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_key)

    zhipu_key = os.getenv("ZHIPU_API_KEY")
    if zhipu_key:
        # 智谱提供 OpenAI 兼容协议，可直接复用 OpenAIEmbeddings
        return OpenAIEmbeddings(
            model="embedding-3",
            api_key=zhipu_key,
            base_url="https://open.bigmodel.cn/api/paas/v4/",
        )

    raise RuntimeError(
        "缺少 Embedding 凭据：请在 .env 中配置 OPENAI_API_KEY 或 ZHIPU_API_KEY"
    )


class KnowledgeBase:
    """个人知识库管理器"""

    def __init__(self, persist_dir: str = "./my_knowledge_db"):
        self.persist_dir = persist_dir
        self.embeddings = _build_embeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", "！", "？", ". ", "! ", "? ", " "],
        )
        self.vector_store = None
        self._load_or_create()

    def _load_or_create(self):
        """加载已有向量库或创建新的"""
        if os.path.exists(self.persist_dir) and os.listdir(self.persist_dir):
            self.vector_store = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings,
            )
            print(f"已加载向量库：{self.vector_store._collection.count()} 个文档块")
        else:
            self.vector_store = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings,
            )
            print("已创建新的向量库")

    def add_file(self, file_path: str) -> int:
        """添加文件到知识库，返回添加的文档块数"""
        loader = TextLoader(file_path=file_path, encoding="utf-8")
        docs = loader.load()

        # 添加文件来源元数据
        for doc in docs:
            doc.metadata["source"] = Path(file_path).name

        chunks = self.text_splitter.split_documents(docs)
        self.vector_store.add_documents(chunks)
        print(f"已添加 {Path(file_path).name}：{len(chunks)} 个文档块")
        return len(chunks)

    def add_text(self, text: str, source: str = "手动添加") -> int:
        """直接添加文本到知识库"""
        chunks = self.text_splitter.create_documents(
            [text], metadatas=[{"source": source}]
        )
        self.vector_store.add_documents(chunks)
        return len(chunks)

    def search(self, query: str, k: int = 3) -> list:
        """搜索知识库"""
        return self.vector_store.similarity_search(query, k=k)

    def get_retriever(self):
        """获取检索器"""
        return self.vector_store.as_retriever(search_kwargs={"k": 3})


# ========== 创建知识库并添加示例数据 ==========

kb = KnowledgeBase("./my_knowledge_db")

# 添加一些示例知识
kb.add_text(
    "菜鸟教程 RUNOOB 的 Python3 基础教程包含以下章节："
    "1. Python 简介与环境搭建 2. 基本数据类型 3. 运算符与表达式 "
    "4. 条件判断 if-else 5. 循环 for/while 6. 函数定义与调用 "
    "7. 模块与包 8. 文件操作 9. 异常处理 10. 面向对象编程",
    source="Python3 教程大纲",
)

kb.add_text(
    "要成为一名优秀的 Python 开发者，建议按以下路线学习："
    "第一步，掌握 Python 基础语法（1-2 周）；"
    "第二步，学习数据结构和算法基础（2-3 周）；"
    "第三步，选择一个方向深入学习（Web 开发/数据分析/AI）；"
    "第四步，做 2-3 个实战项目巩固知识。",
    source="Python 学习路线",
)

kb.add_text(
    "菜鸟教程的在线编程环境支持 Python、JavaScript、Java、C++ 等多种语言。"
    "用户无需安装任何软件，打开浏览器即可编写和运行代码。"
    "在线环境还支持代码高亮、自动补全和错误提示功能。",
    source="在线编程环境说明",
)


# ========== 创建 RAG Agent ==========

@tool
def search_knowledge(query: str) -> str:
    """在个人知识库中搜索相关位置，搜索时使用完整的问题或关键短语

    Args:
        query: 搜索问题或关键短语
    """
    docs = kb.search(query, k=3)
    if not docs:
        return "知识库中未找到相关信息。"

    results = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "未知来源")
        content = doc.page_content[:200]
        results.append(f"[{i}] 来源：{source}\n{content}")

    return "\n\n---\n\n".join(results)


model = init_chat_model("deepseek:deepseek-v4-flash", temperature=0)
agent = create_agent(
    model=model,
    tools=[search_knowledge],
    system_prompt="""
    你是个人知识库助手。
        ## 规则
        1. 所有问题必须先用 search_knowledge 工具检索知识库
        2. 回答时注明信息来源（文档名称）
        3. 如果知识库中没有相关内容，如实告知
        4. 回答要结构化，使用数字列表或分段
    """,
)


# ========== 测试 ==========
def _safe_print(text: str) -> None:
    """在 Windows GBK 控制台下安全打印（忽略无法编码的字符，如 emoji）。"""
    try:
        print(text)
    except UnicodeEncodeError:
        encoding = getattr(getattr(__import__("sys"), "stdout"), "encoding", None) or "utf-8"
        print(text.encode(encoding, errors="replace").decode(encoding, errors="replace"))


def ask(question: str):
    """提问并显示回答"""
    print(f"\n{'=' * 60}")
    print(f"Q: {question}")
    print(f"{'=' * 60}")

    result = agent.invoke({
        "messages": [HumanMessage(content=question)],
    })

    # 显示检索到的内容
    for msg in result["messages"]:
        if msg.type == "tool":
            print(f"\n[检索到的内容]")
            _safe_print(msg.content[:300])

    print(f"\n[回答]")
    _safe_print(result["messages"][-1].content)


ask("Python3 基础教程包含哪些章节？")
ask("如何规划 Python 学习路线？")
ask("菜鸟教程的在线编程环境支持哪些功能？")



"""
输出结果：

已加载向量库：3 个文档块

============================================================
Q: Python3 基础教程包含哪些章节？
============================================================

[检索到的内容]
[1] 来源：Python3 教程大纲
菜鸟教程 RUNOOB 的 Python3 基础教程包含以下章节：1. Python 简介与环境搭建 2. 基本数据类型 3. 运算符与表达式 4. 条件判断 if-else 5. 循环 for/while 6. 函数定义与调用 7. 模块与包 8. 文件操作 9. 异常处理 10. 面向对象编程

---

[2] 来源：Python3 教程大纲
菜鸟教程 RUNOOB 的 Python3 基础教程包含以下章节：1. Python 简介与环境搭建 2. 基本数据类型 3. 运算符与表达式 4. 条件判断 if-else 5. 循环 for/while 

[检索到的内容]
[1] 来源：Python3 教程大纲
菜鸟教程 RUNOOB 的 Python3 基础教程包含以下章节：1. Python 简介与环境搭建 2. 基本数据类型 3. 运算符与表达式 4. 条件判断 if-else 5. 循环 for/while 6. 函数定义与调用 7. 模块与包 8. 文件操作 9. 异常处理 10. 面向对象编程

---

[2] 来源：Python3 教程大纲
菜鸟教程 RUNOOB 的 Python3 基础教程包含以下章节：1. Python 简介与环境搭建 2. 基本数据类型 3. 运算符与表达式 4. 条件判断 if-else 5. 循环 for/while 

[回答]
根据知识库中的《Python3 教程大纲》文档，菜鸟教程（RUNOOB）的 **Python3 基础教程** 包含以下 **10 个章节**：

1. **Python 简介与环境搭建**
2. **基本数据类型**
3. **运算符与表达式**
4. **条件判断 if-else**
5. **循环 for/while**
6. **函数定义与调用**
7. **模块与包**
8. **文件操作**
9. **异常处理**
10. **面向对象编程**

---

**补充说明：**
- 以上信息来源于文档《Python3 教程大纲》。
- 知识库中还提到，该在线编程环境支持代码高亮、自动补全和错误提示功能（来源：《在线编程环境说明》）。
- 此外，根据《Python 学习路线》建议，掌握基础语法（对应上述章节）大约需要 1-2 周时间，之后再学习数据结构和算法基础，并选择一个方向（Web 开发/数据分析/AI）深入。

============================================================
Q: 如何规划 Python 学习路线？
============================================================

[检索到的内容]
[1] 来源：Python 学习路线
要成为一名优秀的 Python 开发者，建议按以下路线学习：第一步，掌握 Python 基础语法（1-2 周）；第二步，学习数据结构和算法基础（2-3 周）；第三步，选择一个方向深入学习（Web 开发/数据分析/AI）；第四步，做 2-3 个实战项目巩固知识。

---

[2] 来源：Python 学习路线
要成为一名优秀的 Python 开发者，建议按以下路线学习：第一步，掌握 Python 基础语法（1-2 周）；第二步，学习数据结构和算法基础（2-3 周）；第三步，选择一个方向深入学习（Web 开发/数据分析/AI）；第四步，做 2-3 个实战项目

[检索到的内容]
[1] 来源：Python3 教程大纲
菜鸟教程 RUNOOB 的 Python3 基础教程包含以下章节：1. Python 简介与环境搭建 2. 基本数据类型 3. 运算符与表达式 4. 条件判断 if-else 5. 循环 for/while 6. 函数定义与调用 7. 模块与包 8. 文件操作 9. 异常处理 10. 面向对象编程

---

[2] 来源：Python3 教程大纲
菜鸟教程 RUNOOB 的 Python3 基础教程包含以下章节：1. Python 简介与环境搭建 2. 基本数据类型 3. 运算符与表达式 4. 条件判断 if-else 5. 循环 for/while 

[回答]
根据知识库中的内容，我为你整理了 Python 学习路线的规划建议。信息来源如下：

## 📚 核心学习路线（来源：*Python 学习路线*）

知识库给出了四步走的核心路线：

1. **掌握 Python 基础语法**（约 1-2 周）
   - 变量、数据类型、运算符、条件判断、循环、函数等

2. **学习数据结构和算法基础**（约 2-3 周）
   - 列表、字典、栈、队列等常用结构及基础算法

3. **选择一个方向深入学习**
   - **Web 开发**（如 Flask/Django）
   - **数据分析**（如 Pandas/NumPy）
   - **AI/机器学习**（如 TensorFlow/PyTorch）

4. **做 2-3 个实战项目巩固知识**
   - 通过真实项目把前面所学融会贯通

---

## 📖 基础语法覆盖内容（来源：*Python3 教程大纲*）

在第一步学基础语法时，建议覆盖以下 10 个章节：

1. Python 简介与环境搭建
2. 基本数据类型
3. 运算符与表达式
4. 条件判断 if-else
5. 循环 for/while
6. 函数定义与调用
7. 模块与包
8. 文件操作
9. 异常处理
10. 面向对象编程

---

## 💻 辅助工具建议（来源：*在线编程环境说明*）

- 学习初期**无需安装任何软件**，可直接使用在线编程环境（如菜鸟教程在线编辑器）
- 支持 Python、JavaScript、Java、C++ 等多种语言
- 具备代码高亮、自动补全和错误提示功能，方便边学边练

---

**总结建议**：先按大纲系统学习基础语法 → 巩固数据结构和算法 → 选定感兴趣的方向深耕 → 用实战项目检验成果。整个路线大约需要 **1-2 个月**可以完成基础部分，具体深度取决于你的目标方向。

如果你能告诉我你更倾向于哪个方向（Web/数据分析/AI），我可以进一步为你检索更详细的资料。

============================================================
Q: 菜鸟教程的在线编程环境支持哪些功能？
============================================================

[检索到的内容]
[1] 来源：在线编程环境说明
菜鸟教程的在线编程环境支持 Python、JavaScript、Java、C++ 等多种语言。用户无需安装任何软件，打开浏览器即可编写和运行代码。在线环境还支持代码高亮、自动补全和错误提示功能。

---

[2] 来源：在线编程环境说明
菜鸟教程的在线编程环境支持 Python、JavaScript、Java、C++ 等多种语言。用户无需安装任何软件，打开浏览器即可编写和运行代码。在线环境还支持代码高亮、自动补全和错误提示功能。

---

[3] 来源：Python3 教程大纲
菜鸟教程 RUNOOB 的 Python3 基础教程包含以下章节：1. Pyth

[检索到的内容]
[1] 来源：在线编程环境说明
菜鸟教程的在线编程环境支持 Python、JavaScript、Java、C++ 等多种语言。用户无需安装任何软件，打开浏览器即可编写和运行代码。在线环境还支持代码高亮、自动补全和错误提示功能。

---

[2] 来源：在线编程环境说明
菜鸟教程的在线编程环境支持 Python、JavaScript、Java、C++ 等多种语言。用户无需安装任何软件，打开浏览器即可编写和运行代码。在线环境还支持代码高亮、自动补全和错误提示功能。

---

[3] 来源：Python3 教程大纲
菜鸟教程 RUNOOB 的 Python3 基础教程包含以下章节：1. Pyth

[回答]
根据知识库中《在线编程环境说明》文档的内容，菜鸟教程的在线编程环境支持以下功能：

1. **多语言支持**：支持 Python、JavaScript、Java、C++ 等多种编程语言。

2. **免安装使用**：用户无需安装任何软件，打开浏览器即可编写和运行代码。

3. **代码高亮**：帮助区分关键字、变量、字符串等代码元素，提高代码可读性。

4. **自动补全**：提供智能代码提示，加快编写速度并减少拼写错误。

5. **错误提示**：在编写或运行时给出错误提示，方便及时排查问题。

---

**信息来源**：《在线编程环境说明》

另外，知识库中还有一份《Python3 教程大纲》，列出了菜鸟教程 Python3 基础教程的章节结构（包括环境搭建、数据类型、运算符、条件判断、循环、函数、模块、文件操作、异常处理、面向对象等），如果您对 Python 教程的内容感兴趣，也可以进一步查阅。

"""
