"""Interactive LangGraph arithmetic agent with persisted human tool approval."""

import json
import os
from dataclasses import dataclass
from typing import Any, Literal
from uuid import uuid4

from dotenv import find_dotenv, load_dotenv
from langchain.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain.tools import ToolRuntime, tool
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_openai.chat_models import ChatOpenAI
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime
from langgraph.types import Command, Interrupt, interrupt
from typing_extensions import Annotated, NotRequired, TypedDict

load_dotenv(find_dotenv())

# Bind the tools once so every model response can either answer or request work.
model = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4.1-mini",
    stream_usage=True,
)


@dataclass(frozen=True)
class UserContext:
    user_id: str


@tool
def multiply(a: int, b: int, runtime: ToolRuntime[UserContext]) -> int:
    """Multiply two numbers.

    Args:
        a: First number.
        b: Second number.
    """
    print(f"Executing multiply for user: {runtime.context.user_id}")
    return a * b


@tool
def add(a: int, b: int, runtime: ToolRuntime[UserContext]) -> int:
    """Add two numbers.

    Args:
        a: First number.
        b: Second number.
    """
    print(f"Executing add for user: {runtime.context.user_id}")
    return a + b


@tool
def divide(a: int, b: int, runtime: ToolRuntime[UserContext]) -> float:
    """Divide one number by another.

    Args:
        a: Dividend.
        b: Divisor.
    """
    print(f"Executing divide for user: {runtime.context.user_id}")
    return a / b


@tool
def subtract(a: int, b: int, runtime: ToolRuntime[UserContext]) -> int:
    """Subtract the second number from the first.

    Args:
        a: First number.
        b: Second number.
    """
    print(f"Executing subtract for user: {runtime.context.user_id}")
    return a - b


tools = [add, multiply, divide, subtract]
model_with_tools = model.bind_tools(tools)

# Only tools listed here are sent through the human-review node. Keeping every
# tool in the set makes this example approval-first; remove a name to let that
# tool route directly to ``tool_node``.
TOOLS_REQUIRING_APPROVAL = {tool.name for tool in tools}
DEFAULT_REJECTION_MESSAGE = (
    "The human reviewer rejected this tool call. Do not retry it."
)


def execute_reviewed_tool_call(request: ToolCallRequest, execute: Any) -> Any:
    """Execute an approved call or return feedback for a rejected call."""

    tool_call = request.tool_call
    review = request.state.get("tool_reviews", {}).get(tool_call["id"], {})

    if review.get("action") == "reject":
        # ToolNode still receives rejected calls so the conversation gets the
        # ToolMessage required to resolve each AI tool-call ID.
        return ToolMessage(
            content=review.get(
                "message",
                DEFAULT_REJECTION_MESSAGE,
            ),
            name=tool_call["name"],
            tool_call_id=tool_call["id"],
            status="error",
        )

    return execute(request)


class AgentState(TypedDict):
    # add_messages appends new messages and replaces messages with matching IDs.
    messages: Annotated[list[AnyMessage], add_messages]
    # Rejection decisions are keyed by tool-call ID for the ToolNode wrapper.
    tool_reviews: NotRequired[dict[str, dict[str, Any]]]


def llm_call(state: AgentState) -> AgentState:
    """Let the LLM decide whether it needs a tool or can answer directly."""

    # The full message history includes prior ToolMessages, allowing the model
    # to turn tool results into a final answer or request another calculation.
    response = model_with_tools.invoke(
        [
            SystemMessage(
                content=(
                    "You are a helpful assistant tasked with performing arithmetic "
                    "on a set of inputs. Use the available arithmetic tools and "
                    "provide an accurate answer based on the tool output."
                )
            )
        ]
        + state["messages"]
    )

    return AgentState(
        messages=[response],
        # Decisions belong only to the previous group of tool calls.
        tool_reviews={},
    )


def should_continue(
    state: AgentState,
) -> Literal["human_review", "tool_node", "__end__"]:
    """Review protected tool calls, execute other tool calls, or finish."""

    tool_calls = state["messages"][-1].tool_calls
    # A mixed batch is reviewed if even one call is protected. Human review
    # collects decisions only for the protected calls; the rest remain intact.
    if any(tool_call["name"] in TOOLS_REQUIRING_APPROVAL for tool_call in tool_calls):
        return "human_review"
    if tool_calls:
        return "tool_node"

    return "__end__"


def human_review(
    state: AgentState,
    runtime: Runtime[UserContext],
) -> Command[Literal["llm_call", "tool_node"]]:
    """Review calls or ask the LLM to regenerate them from human feedback."""

    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage):
        raise TypeError("Human review requires the latest message to be an AIMessage.")

    calls_to_review = [
        tool_call
        for tool_call in last_message.tool_calls
        if tool_call["name"] in TOOLS_REQUIRING_APPROVAL
    ]

    # interrupt() checkpoints the graph here. On resume, it evaluates to the
    # decision payload collected by the CLI instead of interrupting again.
    review_response = interrupt(
        {
            "type": "tool_approval",
            "user_id": runtime.context.user_id,
            "tool_calls": [
                {
                    "tool_call_id": tool_call["id"],
                    "tool_name": tool_call["name"],
                    "arguments": tool_call["args"],
                    "question": f"Approve execution of {tool_call['name']}?",
                    "allowed_decisions": ["approve", "edit", "reject"],
                }
                for tool_call in calls_to_review
            ],
        }
    )
    decisions = review_response.get("decisions")
    if decisions is None and len(calls_to_review) == 1:
        # Also accept the original single-call resume shape.
        decisions = [review_response]

    if not isinstance(decisions, list) or len(decisions) != len(calls_to_review):
        raise ValueError("A decision is required for every reviewed tool call.")

    reviewed_calls = list(zip(calls_to_review, decisions, strict=True))

    edit_requests = [
        (tool_call, decision)
        for tool_call, decision in reviewed_calls
        if decision.get("action") == "edit"
    ]
    if edit_requests:
        feedback_lines: list[str] = []
        for tool_call, decision in edit_requests:
            feedback = decision.get("feedback")
            if not isinstance(feedback, str) or not feedback.strip():
                raise ValueError("Human edit feedback cannot be empty.")
            feedback_lines.append(
                f"- Proposed {tool_call['name']}({json.dumps(tool_call['args'])}): "
                f"{feedback.strip()}"
            )

        # Every tool-call ID must receive a ToolMessage before the model can
        # propose a corrected batch, including calls that did not need review.
        cancelled_tool_messages = [
            ToolMessage(
                content=(
                    "This proposed tool call was not executed because the human "
                    "reviewer requested a revised tool call."
                ),
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
                status="error",
            )
            for tool_call in last_message.tool_calls
        ]
        revision_message = HumanMessage(
            content=(
                "The previous tool calls were not executed. Apply the human "
                "reviewer's corrections below, then create new corrected tool "
                "calls. Do not answer the arithmetic directly; use the tools.\n"
                + "\n".join(feedback_lines)
            )
        )
        return Command(
            update={
                "messages": [*cancelled_tool_messages, revision_message],
                "tool_reviews": {},
            },
            goto="llm_call",
        )

    tool_reviews: dict[str, dict[str, Any]] = {}

    for tool_call, decision in reviewed_calls:
        action = decision.get("action")
        if action == "reject":
            tool_reviews[tool_call["id"]] = decision
        elif action != "approve":
            raise ValueError(f"Unknown human review action: {action!r}")

    # Approved calls need no metadata. Rejected calls are intercepted by
    # execute_reviewed_tool_call while all other calls execute normally.
    return Command(
        update={"tool_reviews": tool_reviews},
        goto="tool_node",
    )


# Build the cycle: model -> optional review -> tools -> model. A model response
# without tool calls follows the END branch and completes the turn.
agent_builder = StateGraph(state_schema=AgentState, context_schema=UserContext)
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("human_review", human_review)
agent_builder.add_node(
    "tool_node",
    ToolNode(tools=tools, wrap_tool_call=execute_reviewed_tool_call),
)
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call", should_continue, ["human_review", "tool_node", END]
)
agent_builder.add_edge("tool_node", "llm_call")


def stream_graph(
    agent: Any,
    graph_input: AgentState | Command,
    run_config: dict[str, Any],
    user_context: UserContext,
) -> None:
    """Run the graph until it completes or reaches an interrupt."""

    events = agent.stream(
        input=graph_input,
        stream_mode=["messages", "updates"],
        version="v2",
        config=run_config,
        context=user_context,
    )

    for event in events:
        if event["type"] == "messages":
            # Token chunks make the assistant response appear incrementally.
            chunk, _metadata = event["data"]
            if chunk.content:
                print(chunk.content, end="", flush=True)
        elif event["type"] == "updates":
            updates = event["data"]
            # Interrupts are handled by the checkpoint/resume loop below.
            if "__interrupt__" not in updates:
                print("\nState update:", updates)


def get_pending_interrupts(agent: Any, run_config: dict[str, Any]) -> list[Interrupt]:
    """Read interrupts saved on the current thread checkpoint."""

    # A snapshot can contain multiple interrupted tasks if the graph later
    # grows parallel branches, so flatten every task's interrupt collection.
    snapshot = agent.get_state(run_config)
    return [
        pending_interrupt
        for task in snapshot.tasks
        for pending_interrupt in task.interrupts
    ]


def prompt_for_tool_decision(request: dict[str, Any]) -> dict[str, Any]:
    """Collect an approval decision for one proposed tool call."""

    print("\n\nHuman approval required")
    print(f"Tool: {request['tool_name']}")
    print("Arguments:", json.dumps(request["arguments"], indent=2))

    while True:
        choice = input("[a]pprove, [e]dit, or [r]eject? ").strip().lower()

        if choice in {"a", "approve"}:
            return {"action": "approve"}

        if choice in {"e", "edit"}:
            feedback = input("Describe how the tool call should change: ").strip()
            if not feedback:
                print("Please describe the correction.")
                continue

            return {"action": "edit", "feedback": feedback}

        if choice in {"r", "reject"}:
            message = input("Optional rejection reason: ").strip()
            return {
                "action": "reject",
                "message": message or DEFAULT_REJECTION_MESSAGE,
            }

        print("Please enter a, e, or r.")


def prompt_for_decision(pending_interrupt: Interrupt) -> dict[str, Any]:
    """Collect decisions for every tool call in one graph interrupt."""

    request = pending_interrupt.value
    tool_calls = request.get("tool_calls")

    if not isinstance(tool_calls, list):
        # Compatibility with checkpoints created by the earlier implementation.
        return prompt_for_tool_decision(request)

    return {
        "decisions": [
            prompt_for_tool_decision(tool_request) for tool_request in tool_calls
        ]
    }


def build_resume_command(pending_interrupts: list[Interrupt]) -> Command:
    """Collect decisions and map them to one or more interrupt IDs."""

    decisions = {
        pending_interrupt.id: prompt_for_decision(pending_interrupt)
        for pending_interrupt in pending_interrupts
    }

    if len(pending_interrupts) == 1:
        # LangGraph accepts a direct resume value for one interrupt.
        return Command(resume=next(iter(decisions.values())))

    # Multiple interrupts must be resumed by their checkpointed interrupt IDs.
    return Command(resume=decisions)


def print_usage(
    usage_callback: UsageMetadataCallbackHandler,
    *,
    execution_id: str,
    user_id: str,
    thread_id: str,
) -> None:
    """Print aggregate usage for one complete user turn."""

    usage_by_model = usage_callback.usage_metadata
    usage_record = {
        "execution_id": execution_id,
        "user_id": user_id,
        "thread_id": thread_id,
        "input_tokens": sum(
            usage.get("input_tokens", 0) for usage in usage_by_model.values()
        ),
        "output_tokens": sum(
            usage.get("output_tokens", 0) for usage in usage_by_model.values()
        ),
        "total_tokens": sum(
            usage.get("total_tokens", 0) for usage in usage_by_model.values()
        ),
        "usage_by_model": usage_by_model,
    }

    print("\nUsage:", usage_record)


def main() -> None:
    print("LangGraph Agent.")
    print("Type 'exit' to quit.\n")

    # The context is available to graph nodes and injected into every tool via
    # ToolRuntime. Reusing the thread ID preserves one conversation checkpoint.
    user_context = UserContext(user_id="abcd1234")
    thread_id = "conversation_abcd1234"
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise RuntimeError(
            "DATABASE_URL is required. Set it to a PostgreSQL URL for the "
            "invoice database, for example: "
            "postgresql://postgres:<password>@localhost:5432/invoice?sslmode=disable"
        )

    with PostgresSaver.from_conn_string(database_url) as checkpointer:
        # setup() creates checkpoint tables when they do not already exist.
        checkpointer.setup()
        agent = agent_builder.compile(checkpointer=checkpointer)

        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in {"exit", "quit"}:
                break

            execution_id = str(uuid4())
            usage_callback = UsageMetadataCallbackHandler()
            # Config carries persistence identity, token tracking, and metadata
            # for the entire graph run, including any interrupt resumptions.
            run_config = {
                "configurable": {"thread_id": thread_id},
                "callbacks": [usage_callback],
                "metadata": {
                    "execution_id": execution_id,
                    "user_id": user_context.user_id,
                },
            }

            graph_input: AgentState | Command = AgentState(
                messages=[HumanMessage(content=user_input)]
            )

            # Each interrupt pauses the stream. The saved checkpoint is read,
            # decisions are collected, and Command(resume=...) continues it.
            while True:
                stream_graph(agent, graph_input, run_config, user_context)
                pending_interrupts = get_pending_interrupts(agent, run_config)

                if not pending_interrupts:
                    break

                graph_input = build_resume_command(pending_interrupts)

            print_usage(
                usage_callback,
                execution_id=execution_id,
                user_id=user_context.user_id,
                thread_id=thread_id,
            )


if __name__ == "__main__":
    main()
