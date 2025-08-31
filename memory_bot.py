from typing import TypedDict, List, Union, Annotated

from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

import config


class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]


llm = ChatOpenAI(api_key=config.OPENAI_API_KEY, model="gpt-4o-mini")


def process(state: AgentState) -> AgentState:
    """This node will send llm request"""
    response = llm.invoke(input=state["messages"])
    state["messages"].append(AIMessage(content=response.content))
    return state


graph = StateGraph(state_schema=AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)

agent = graph.compile()

chat_history = []

user_input = input("You: ")

while user_input != "exit":
    chat_history.append(HumanMessage(content=user_input))
    state: AgentState = Annotated[
        AgentState, agent.invoke(input={"messages": chat_history})
    ]
    chat_history = state["messages"]
    print(f"AI: {state['messages'][-1].content}")
    user_input = input("You: ")
