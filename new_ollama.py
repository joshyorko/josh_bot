from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_community.llms.ollama import Ollama
model = OllamaFunctions(model="mistral",llm=Ollama)


model = model.bind(
    functions=[
        {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, " "e.g. San Francisco, CA",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                    },
                },
                "required": ["location"],
            },
        }
    ],
    function_call={"name": "get_current_weather"},
)


from langchain.schema import HumanMessage

messages = [HumanMessage(content="What is the weather in Boston?")]

model.invoke(messages)