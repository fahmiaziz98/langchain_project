import os
from langchain_cohere import ChatCohere
from langchain import hub
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain_core.prompts import ChatPromptTemplate
# from langchain_cohere.react_multi_hop.agent import create_cohere_react_agent

from llm_api.src.chains.review_chain import setup_vector_chain
from llm_api.src.chains.cyper_chain import create_cypher_qa_chain
from llm_api.src.tools.wait_times import get_current_wait_times, get_most_available_hospital
from dotenv import load_dotenv, find_dotenv

from langchain_groq import ChatGroq

load_dotenv(find_dotenv())
AGENT_MODEL = os.getenv("AGENT_MODEL")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HOSPITAL_AGENT_MODEL = os.getenv("HOSPITAL_AGENT_MODEL")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

reviews_vector_chain = setup_vector_chain()
hospital_cypher_chain = create_cypher_qa_chain()

tools = [
    Tool(
        name="Experiences",
        func=reviews_vector_chain.invoke,
        description="""Use this tool to answer questions related to patient 
        experiences, reviews, feelings, or any other qualitative information about a patient. 
        This tool leverages semantic search to provide insights from reviews. 

        Note:
        - This tool is NOT suitable for answering objective questions that involve counting, percentages, aggregations, or listing factual data.
        - Provide the entire question as input to the tool. For example, for the question "Are patients satisfied with their care?", the input should be exactly "Are patients satisfied with their care?".
        """,
    ),
    Tool(
        name="Graph",
        func=hospital_cypher_chain.invoke,
        description="""Use this tool to answer questions about patients, physicians, hospitals, insurance payers, patient review statistics, and hospital visit details. 

        Note:
        - Provide the entire question as input to the tool. For example, if the question is "How many visits have there been?", the input should be exactly "How many visits have there been?".
        """,
    ),

    Tool(
        name="Waits",
        func=get_current_wait_times,
        description="""Use when asked about current wait times
        at a specific hospital. This tool can only get the current
        wait time at a hospital and does not have any information about
        aggregate or historical wait times. Do not pass the word "hospital"
        as input, only the hospital name itself. For example, if the prompt
        is "What is the current wait time at Jordan Inc Hospital?", the
        input should be "Jordan Inc".
        """,
    ),
    Tool(
        name="Availability",
        func=get_most_available_hospital,
        description="""
        Use when you need to find out which hospital has the shortest
        wait time. This tool does not have any information about aggregate
        or historical wait times. This tool returns a dictionary with the
        hospital name as the key and the wait time in minutes as the value.
        """,
    ),
]

def agent_executor() -> AgentExecutor:
    """
    This function returns an AgentExecutor object to be used to interact with the hospital agent.
    
    The AgentExecutor is a tool from LangChain that handles the interactions with the agent. 
    It takes the agent, tools, and other parameters and returns an object that can be used to 
    interact with the agent. The object has a single method, invoke, which takes a dictionary
    with a single key, 'input', and returns a dictionary with the response.
    
    The agent executor created by this function is configured to use the hospital agent, 
    which is a chatbot that responds to questions about hospitals, patients, reviews, etc. 
    The agent is created with a temperature of 0.1, which is a measure of how creative the
    agent is. A higher temperature means that the agent will generate more creative and
    possibly incorrect responses. A lower temperature means that the agent will generate more
    accurate and predictable responses.
    """
    # Create the chat model with a temperature of 0.1.
    chat_model = ChatGroq(
            model=AGENT_MODEL,
            api_key=GROQ_API_KEY,
            temperature=0.1,
            streaming=True,
        )
   
    # Create the chat agent with the tools and the chat model.
    rag_agent = create_react_agent(
        llm=chat_model,
        prompt=hub.pull("hwchase17/react"),
        tools=tools,
    )

    # Create the AgentExecutor with the chat agent and the tools.
    return AgentExecutor(
        agent=rag_agent,
        tools=tools,
        return_intermediate_steps=True,
        verbose=True,
    )

if __name__ == "__main__":
    agent = agent_executor()