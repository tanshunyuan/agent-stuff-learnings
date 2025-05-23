from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AnyMessage
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel

# Checkpoint is a snapshot of the State
# whereas State is the live and mutable data at the current moment
checkpointer = InMemorySaver()

# https://python.langchain.com/api_reference/langchain/chat_models/langchain.chat_models.base.init_chat_model.html
model = init_chat_model(
    "qwen2.5:7b",  # Must be installed on my local!
    model_provider="ollama",
    temperature=0,
)


class WeatherResponse(BaseModel):
    conditions: str


def prompt(state: AgentState, config: RunnableConfig) -> list[AnyMessage]:
    # get username from agent.invoke `config` argument
    user_name = config["configurable"].get("user_name")
    system_msg = f"You are a helpful assistant. Address the user as {user_name}."
    result = [{"role": "system", "content": system_msg}] + state["messages"]
    print(f"result: {result} \n")
    return result


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


agent = create_react_agent(
    model=model,
    tools=[get_weather],
    prompt=prompt,
    checkpointer=checkpointer,
    # Structured output requires an additional call to the LLM to format the response according to the schema.
    # Can be accessed via ['structured_response']
    response_format=WeatherResponse,
)

# when thread_id is shared across agents, the orignal message history is included in the second invocation
config = {"configurable": {"user_name": "John Smith", "thread_id": "1"}}

# Run the agent
sf_response = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]},
    config=config,
)

ny_response = agent.invoke(
    {"messages": [{"role": "user", "content": "what about new york?"}]}, config=config
)

print(f"sf_response: {sf_response['structured_response']} \n")
print(f"ny_response: {ny_response['structured_response']} \n")
