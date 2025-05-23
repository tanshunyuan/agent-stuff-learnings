# https://langchain-ai.github.io/langgraph/tutorials/get-started/1-build-basic-chatbot/

from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from langchain_core.messages import ToolMessage

# by default, it's recognised as a tool
from langchain_tavily import TavilySearch

from pprint import pprint
from dotenv import load_dotenv

import os
import json

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

llm_with_tools = llm.bind_tools(tools)


# Graph !== llm, the graph is just a outline of what components to call,
# the nodes will then invoke the llm
def chatbot_node(state: State):
    # return {"messages": [llm.invoke(state["messages"])]}
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __init__(self, tools: list):
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):  # if there are messages
            message = messages[-1]
        else:
            raise ValueError("No message found in input")

        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}


# tells the llm what tools to call based on the last message
def route_tools(state: State):
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")

    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END


tool_node = BasicToolNode(tools=tools)

graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot_node)
graph_builder.add_node("tools", tool_node)

# sugar syntax for graph_builder.add_edge(START, "chatbot")
graph_builder.set_entry_point("chatbot")

graph_builder.add_conditional_edges(
    "chatbot", route_tools, {"tools": "tools", END: END}
)

graph_builder.add_edge("tools", "chatbot")
# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge(START, "chatbot")

graph = graph_builder.compile()


# Stream updates from the graph
def stream_graph_updates(user_input: str):
    # this feels a bit magic, are "messages", "role" and "content" fixed? No, but this follows the State class schema
    # messages: Annotated[list, add_messages], add_messages is the reducer
    streams = graph.stream({"messages": [{"role": "user", "content": user_input}]})
    for event in streams:
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)


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
