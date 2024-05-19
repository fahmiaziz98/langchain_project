import logging
from typing import List

from dotenv import find_dotenv, load_dotenv
from langchain.agents import AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_cohere.react_multi_hop.agent import create_cohere_react_agent

from src.tools import get_tools, Tool
from src.models import get_chat_model

# Load the environment variables from my .env file
load_dotenv(find_dotenv())
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_react_agent_executor(
    llm: BaseChatModel,
    tools: List[Tool],
) -> AgentExecutor:
    """
    Create an AgentExecutor instance with the given language model and tools.

    Parameters:
    - llm: An instance of the BaseChatModel class representing the language model.
    - tools: A list of instances of the Tool class representing the tools.

    Returns:
    - An instance of the AgentExecutor class.
    """
    prompt = ChatPromptTemplate.from_template("{input}")
    
    agent = create_cohere_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt,
    )
    
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True  # Enable verbose mode for logging
    )
    
    return agent_executor


def run_agent(input: str, vector):
    """
    Run the agent with the given input and vector.

    Parameters:
    - input (str): The input prompt for the agent.
    - vector (Any): The vector store retriver for the agent.

    Returns:
    - str: The output of the agent.
    """

    preamble = """
    You are an AI chatbot that helps users chat with PDF documents.
    Use the following pieces of context to answer the question at the end. Please follow the following rules:
    1. If you find the answer, write the answer in a Elegant way and add the list of sources that are **directly** used to derive the answer.
    and if there are none then search the internet using internet_search tools and add the list of sources that are **indirectly** used to derive the answer.

    Example:
    The Answer is derived from[1] this page
    [1] Source_ Page:PageNumber

    {context}

    Question: {question}
    Helpful Answer:"""

    llm = get_chat_model()
    tools = get_tools(vector)

    agent_executor = get_react_agent_executor(llm, tools)

    output = agent_executor.invoke(
        {
            'input': input,
            "preamble": preamble,
        }
    )
    return output["output"]

# if __name__ == "__main__":
#     # text = "List of deductions and exemptions withdrawn under the new tax regime for FY 2020-21 (AY 2021-22)"
#     # text = "when openai launch model GPT-4o"
#     # text = "write python code load qunatization Llama 3"
#     text = "create plot show trend bitcoin from 2014 - 2024"
#     # text = "Search job from upwork for Ai Engineer with level junior"
#     response = run_agent("cohere", "command-r-plus", text)
#     print(response["output"])
