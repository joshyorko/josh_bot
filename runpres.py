from typing import List
import pandas as pd
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.chat_models import ChatOllama
from rich import print as rprint
from prompts import presidents
import requests
from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter
import traceback


BASE_URL = "http://192.168.86.29:11434"

def list_models():
    response = requests.get(f"{BASE_URL}/api/tags")
    if response.status_code == 200:
        models = response.json().get("models", [])
        return [model["name"] for model in models]
    else:
        print(f"Error: {response.status_code}")
        return []



class Term(BaseModel):
    start: str = Field(..., description="The start year of the president's term.")
    end: str = Field(..., description="The end year of the president's term .")


class President(BaseModel):
    name: str = Field(..., description="The name of the president.")
    birth_year: int = Field(..., description="The birth year of the president.")
    death_year: int = Field(..., description="The death year of the president.")
    party: str = Field(..., description="The political party of the president.")
    term: Term = Field(..., description="The term of the president.")
    interesting_fact: str = Field(..., description="An interesting fact about the president.")
    popular_vote: int = Field(..., description="The popular vote of the president.")



def get_presidents(selected_model_name: str):
    model = ChatOllama(model=selected_model_name ,base_url=BASE_URL)
    # query a random president from presidents list
    query = pd.DataFrame(presidents).sample(1).to_string(index=False, header=False)
    rprint(f"US PRESIDENT CHOSEN: {query}")

    parser = JsonOutputParser(pydantic_object=President)

    prompt_template = PromptTemplate(
        template="Answer the user query.\n{format_instructions}\n{query}\n",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt_template | model | parser

    output = chain.invoke({"query": query})

    rprint(output)

    return output

model_name_list = list_models()
model_completer = WordCompleter(model_name_list, ignore_case=True)

model_name = prompt("Select a model: ", completer=model_completer)

rprint(f"TESTING: {model_name}")


try:
    get_presidents(model_name)
    rprint("\n\n\n")
except Exception as e:
    rprint(f"Error: {e}")
    rprint(traceback.format_exc())