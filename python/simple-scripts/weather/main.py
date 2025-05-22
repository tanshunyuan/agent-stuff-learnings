from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent

# https://python.langchain.com/api_reference/langchain/chat_models/langchain.chat_models.base.init_chat_model.html
model = init_chat_model(
  "qwen2.5:7b",
  model_provider="ollama",
  temperature=0
)

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

agent = create_react_agent(
    model=model,
    tools=[get_weather],
    prompt="You are a helpful assistant"
)

# Run the agent
response = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
)

print(response)
