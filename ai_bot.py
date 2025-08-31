from typing import TypedDict, List

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

import config


class AgentState(TypedDict):
    messages: List[HumanMessage]


llm = ChatOpenAI(api_key=config.OPENAI_API_KEY, model="gpt-4o-mini")


def process(state: AgentState) -> AgentState:
    response = llm.invoke(input=state["messages"])
    print(f"Agent: {response.content}")
    return state


graph = StateGraph(state_schema=AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)

agent = graph.compile()

user_input = input("You: ")

while user_input != "exit":
    agent.invoke(input={"messages": [HumanMessage(content=user_input)]})
    user_input = input("You: ")
