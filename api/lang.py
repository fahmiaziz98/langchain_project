import os
from langchain_cohere.chat_models import ChatCohere
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from handler import MyCustomHandler


from dotenv import load_dotenv, find_dotenv
from queue import Queue

load_dotenv(find_dotenv())
COHERE_API_KEY = os.environ["COHERE_API_KEY"]

# Creating a Streamer queue
streamer_queue = Queue()

# Creating an object of custom handler
my_handler = MyCustomHandler(streamer_queue)
def get_response():

    # template = """
    # You are a helpful assistant. Answer the following questions considering the history of the conversation:

    # Chat history: {chat_history}

    # User question: {user_question}
    # """

    # prompt = ChatPromptTemplate.from_template(template)

    llm = ChatCohere(
            model="command-r-plus",
            temperature=0.1,
            cohere_api_key=COHERE_API_KEY,
            streaming=True,
            callbacks=[my_handler]
        )
        
    # chain = prompt | llm 
    
    return llm