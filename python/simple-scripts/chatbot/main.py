# https://langchain-ai.github.io/langgraph/tutorials/get-started/1-build-basic-chatbot/

from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

# by default, it's recognised as a tool
from langchain_tavily import TavilySearch
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool

from langgraph.types import Command, interrupt

from dotenv import load_dotenv

import os


load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


# State is the FIRST thing to define in a graph
class State(TypedDict):
    # Using a list appends incoming messages instead of overwriting the existing messages
    # Docs for add_messages: https://langchain-ai.github.io/langgraph/reference/graphs/#langgraph.graph.message.add_messages
    messages: Annotated[list, add_messages]


web_search_tool = TavilySearch(max_results=2)


@tool
def human_assistance(query: str) -> str:
    """Request assistance from a human."""
    human_response = interrupt({"query": query})
    print(f"human_response: {human_response}")
    return human_response["data"]


tools = [
    # web_search_tool,
    human_assistance
]

llm = init_chat_model(
    "qwen2.5:7b",
    model_provider="ollama",
    temperature=0,
)

# Reassignment is required to use the tools
llm_with_tools = llm.bind_tools(tools)


# Graph !== llm, the graph is just a outline of what components to call,
# the nodes will then invoke the llm
def chatbot_node(state: State):
    # message = llm.invoke(state["messages"])
    # Because we will be interrupting during tool execution,
    # we disable parallel tool calling to avoid repeating any
    # tool invocations when we resume.
    message = llm_with_tools.invoke(state["messages"])
    assert len(message.tool_calls) <= 1
    return {"messages": [message]}


graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot_node)

tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)


graph_builder.add_conditional_edges("chatbot", tools_condition)

# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")

# sugar syntax for graph_builder.add_edge(START, "chatbot")
graph_builder.set_entry_point("chatbot")

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory, interrupt_before=["tools"])

config = {"configurable": {"thread_id": "1"}}


# Stream updates from the graph
def stream_graph_updates(user_input: str):
    # this feels a bit magic, are "messages", "role" and "content" fixed? No, but this follows the State class schema
    # messages: Annotated[list, add_messages], add_messages is the reducer
    events = graph.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config=config,
        stream_mode="values",
    )
    for event in events:
        if "messages" in event:
            event["messages"][-1].pretty_print()

    # todo: figure out how to introduce the human in the loop part
    snapshot = graph.get_state(config=config)


while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        stream_graph_updates(user_input)
    except:
        # fallback if input() is not available
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break
