from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import Ollama


llm = Ollama(
    model="josh_dolphin",
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
)


llm(input("Enter prompt: "))