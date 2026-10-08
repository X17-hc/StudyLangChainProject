from dotenv import load_dotenv
load_dotenv()

import asyncio
from langgraph.graph import END, START, MessagesState, StateGraph

async  def slow_researcher(state: MessagesState):
    last_human = state["messages"][-1].content if state["messages"] else "No task provided."
    await asyncio.sleep(8)
    return {
        "messages": [
            {
                "role": "ai",
                "content": (
                    "[researcher finished after 8s]\\n"
                    f"latest task: {last_human}\\n"
                    "summary: async subagents return a task ID immediately, "
                    "run in the background, and can be checked or updated later."
                ),
            }
        ]
    }

builder = StateGraph(MessagesState)

# 需要先注册边，再添加
builder.add_node("slow_researcher", slow_researcher)

builder.add_edge(START, "slow_researcher")

builder.add_edge("slow_researcher", END)
graph = builder.compile()