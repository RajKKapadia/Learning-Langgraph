from typing import TypedDict, List, Union, Annotated, Sequence

from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langchain_core.tools import tool

import config


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


@tool
def add(a: int, b: int) -> int:
    "This function adds two numbers and return the sum of them."
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    "This function adds two numbers and return the product of them."
    return a * b


tools = [add, multiply]

llm = ChatOpenAI(api_key=config.OPENAI_API_KEY, model="gpt-4o-mini").bind_tools(tools)


def process(state: AgentState) -> AgentState:
    response = llm.invoke(
        [SystemMessage(content="You are a helpful assistant.")] + state["messages"]
    )
    return {"messages": response}


def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"


graph = StateGraph(state_schema=AgentState)
graph.add_node("process", process)
tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)
graph.set_entry_point("process")
graph.add_conditional_edges(
    source="process", path=should_continue, path_map={"continue": "tools", "end": END}
)
graph.add_edge("tools", "process")

agent = graph.compile()


def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()


inputs = {
    "messages": [
        (
            "user",
            "Add 40 + 12 and then multiply the result by 6. Also tell me a joke please.",
        )
    ]
}
print_stream(agent.stream(inputs, stream_mode="values"))
