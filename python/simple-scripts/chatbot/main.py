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

from pprint import pprint
from dotenv import load_dotenv

import os


load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

web_search_tool = TavilySearch(max_results=2)
tools = [web_search_tool]


# State is the FIRST thing to define in a graph
class State(TypedDict):
    # Using a list appends incoming messages instead of overwriting the existing messages
    # Docs for add_messages: https://langchain-ai.github.io/langgraph/reference/graphs/#langgraph.graph.message.add_messages
    messages: Annotated[list, add_messages]


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
    return {"messages": [llm.invoke(state["messages"])]}
    # return {"messages": [llm_with_tools.invoke(state["messages"])]}



graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot_node)

tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)


graph_builder.add_conditional_edges("chatbot", tools_condition)

# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")
# graph_builder.add_edge(START, "chatbot")

# sugar syntax for graph_builder.add_edge(START, "chatbot")
graph_builder.set_entry_point("chatbot")

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "1"}}

# Get graph state
snapshot = graph.get_state(config=config)


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
        event["messages"][-1].pretty_print()
        # for value in event.values():
        #     print("Assistant:", value["messages"][-1].content)

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
