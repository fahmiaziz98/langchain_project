import os
from langchain_cohere import ChatCohere
from langchain.agents import Tool, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_cohere.react_multi_hop.agent import create_cohere_react_agent

from llm_api.src.chains.review_chain import setup_vector_chain
from llm_api.src.chains.cyper_chain import create_cypher_qa_chain
from llm_api.src.tools.wait_times import get_current_wait_times, get_most_available_hospital

HOSPITAL_AGENT_MODEL = os.getenv("HOSPITAL_AGENT_MODEL")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

reviews_vector_chain = setup_vector_chain()
hospital_cypher_chain = create_cypher_qa_chain()

tools = [
    Tool(
        name="Experiences",
        func=reviews_vector_chain.invoke,
        description="""Useful when you need to answer questions
        about patient experiences, feelings, or any other qualitative
        question that could be answered about a patient using semantic
        search. Not useful for answering objective questions that involve
        counting, percentages, aggregations, or listing facts. Use the
        entire prompt as input to the tool. For instance, if the prompt is
        "Are patients satisfied with their care?", the input should be
        "Are patients satisfied with their care?".
        """,
    ),
    Tool(
        name="Graph",
        func=hospital_cypher_chain.invoke,
        description="""Useful for answering questions about patients,
        physicians, hospitals, insurance payers, patient review
        statistics, and hospital visit details. Use the entire prompt as
        input to the tool. For instance, if the prompt is "How many visits
        have there been?", the input should be "How many visits have
        there been?".
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
    chat_model = ChatCohere(
        model=HOSPITAL_AGENT_MODEL,
        temperature=0.1,
        cohere_api_key=COHERE_API_KEY,
        streaming=True,
    )

    # Create the chat agent with the tools and the chat model.
    rag_agent = create_cohere_react_agent(
        llm=chat_model,
        prompt=ChatPromptTemplate.from_template("{input}"),
        tools=tools,
    )

    # Create the AgentExecutor with the chat agent and the tools.
    return AgentExecutor(
        agent=rag_agent,
        tools=tools,
        return_intermediate_steps=True,
        verbose=True,
    )
