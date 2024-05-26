import os
from dotenv import load_dotenv, find_dotenv
from langchain_cohere.chat_models import ChatCohere
from langchain_community.vectorstores import Chroma
from langchain_cohere import CohereEmbeddings
from langchain_cohere.react_multi_hop.agent import create_cohere_react_agent
from langchain.agents import AgentExecutor, Tool

from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate
)

from tools import get_current_wait_time

load_dotenv(find_dotenv())
COHERE_API_KEY = os.environ["COHERE_API_KEY"]
CHROMA_PATH = "db_vector"

review_template_str = """
Your job is to use the provided data about hospitals, patients, payers, physicians, and visits to answer questions.
Use the following context to answer questions. Be as detailed as possible, 
but don't make up any information that's not from the context. If you don't know an answer, say you don't know.

{context}
"""

review_system_prompt = SystemMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["context"],
        template=review_template_str,
    )
)

review_human_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["question"],
        template="{question}",
    )
)

messages = [review_system_prompt, review_human_prompt]
prompt_template = ChatPromptTemplate(
    input_variables=["context", "question"],
    messages=messages,
)

chat_model = ChatCohere(
    model="command-r-plus",
    temperature=0.2,
    cohere_api_key=COHERE_API_KEY,
    streaming=True,
)


vector_db = Chroma(
    embedding_function=CohereEmbeddings(cohere_api_key=COHERE_API_KEY),
    persist_directory=CHROMA_PATH
)

retriver_vector_db = vector_db.as_retriever(k=5)

CHAIN = (
    {"context": retriver_vector_db, "question": RunnablePassthrough()}
    | prompt_template
    | chat_model
    | StrOutputParser()
)

tools = [
    Tool(
        name="qa",
        func=CHAIN.invoke,
        description="""
        Useful when you need to answer questions
        about patient reviews or experiences at the hospital, answering questions about specific visits
        such as payer, pyhsician, patient, visitor, hospital name, or doctor information.
        Submit the entire question as input to this tool.
        """,
    ),
    Tool(
        name="Waits",
        func=get_current_wait_time,
        description="""Use when asked about current wait times
        at a specific hospital. This tool can only get the current
        wait time at a hospital and does not have any information about
        aggregate or historical wait times. This tool returns wait times in
        minutes. Do not pass the word "hospital" as input,
        only the hospital name itself. For instance, if the question is
        "What is the wait time at hospital A?", the input should be "A".
        """,
    ),
]

prompt = ChatPromptTemplate.from_template("{input}")

agent = create_cohere_react_agent(
    llm=chat_model,
    tools=tools,
    prompt=prompt
)
    
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    return_intermediate_steps=True,
    verbose=True  # Enable verbose mode for logging
)


question = "What have patients said about their comfort at the hospital?"
res = agent_executor.invoke(
    {"input": question}
)
print(res["output"])