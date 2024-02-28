from langchain import hub
from langchain.agents import AgentExecutor, create_self_ask_with_search_agent
from langchain_community.llms.ollama import Ollama
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchRun
from langchain_community.callbacks 


search = DuckDuckGoSearchRun(name='search', verbose=True)

# Get the prompt to use - you can modify this!
prompt = hub.pull("hwchase17/self-ask-with-search")


llm = Ollama(model="mistral") 

agent = create_self_ask_with_search_agent(llm, tools, prompt)

# Create an agent executor by passing in the agent and tools
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True,handle_parsing_errors=True)

st_cb = StreamlitCallbackHandler( expand_new_thoughts=False)

agent_executor.invoke(
    {"input": "Who won the women's US open in 2018?"},
)