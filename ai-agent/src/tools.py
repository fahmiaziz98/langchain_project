import os
from typing import List

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.agents import Tool
from langchain_experimental.utilities import PythonREPL
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.utilities.tavily_search import TavilySearchAPIWrapper

from dotenv import load_dotenv, find_dotenv
# get env variables
load_dotenv(find_dotenv())
TAVILY_API_KEY = os.environ["TAVILY_API_KEY"]


def get_tavily_search_tools() -> Tool:
    """
    Returns a tool that searches the internet for a query using the Tavily API.

    Returns:
        TavilySearchResults: A tool that searches the internet for a query using the Tavily API.
    """

    tavilySearchAPIWrapper = TavilySearchAPIWrapper(tavily_api_key=TAVILY_API_KEY)
    internet_search = TavilySearchResults(api_wrapper=tavilySearchAPIWrapper)

    class TavilySearchInput(BaseModel):
        query: str = Field(description='Query to search the internet with')

    internet_search.name = 'internet_search'
    internet_search.description = 'Returns a list of relevant document snippets for a textual query retrieved from the internet.'
    internet_search.args_schema = TavilySearchInput

    return internet_search

def get_python_interpreter_tools() -> Tool:
    """
    Creates a tool that executes python code and returns the result.

    Returns:
        Tool: A tool that executes python code and returns the result.
    """
    python_repl = PythonREPL()
    class ToolInput(BaseModel):
        code: str = Field(description='Python code to execute.')

    repl_tool = Tool(
        name='python_repl',
        description='Executes python code and returns the result. The code runs in a static sandbox without interactive mode, so print output or save output to a file.',
        func=python_repl.run,
    )
    repl_tool.name = 'python_interpreter'
    repl_tool.args_schema = ToolInput

    return repl_tool

def retriver_tools(vector):
    vectorstore_search = create_retriever_tool(
        retriever=vector,
        name="vectorstore_search",
        description="Retrieve relevant info from the vectorstore that contains documents related to the user's query."
    )

    return vectorstore_search

def get_tools(vector) -> List[Tool]:
    """
    Returns a list of tools that can be used by an agent.
    """
    tools = []

    # Create the tools
    internet_search = get_tavily_search_tools()
    # repl_tool = get_python_interpreter_tools()
    retriver = retriver_tools(vector)

    tools.append(internet_search)
    # tools.append(repl_tool)
    tools.append(retriver)

    return tools