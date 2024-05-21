import os
from langchain_core.messages import AIMessage, HumanMessage
from langchain_cohere.chat_models import ChatCohere
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


from dotenv import load_dotenv, find_dotenv


load_dotenv(find_dotenv())
COHERE_API_KEY = os.environ["COHERE_API_KEY"]


def get_response(user_query, chat_history):

    template = """
    You are a helpful assistant. Answer the following questions considering the history of the conversation:

    Chat history: {chat_history}

    User question: {user_question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatCohere(
            model="command-r-plus",
            temperature=0.1,
            cohere_api_key=COHERE_API_KEY,
            streaming=True
        )
        
    chain = prompt | llm | StrOutputParser()
    
    return chain.stream({
        "chat_history": chat_history,
        "user_question": user_query,
    })