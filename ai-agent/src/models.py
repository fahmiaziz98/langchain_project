import os
import logging
from langchain_cohere.chat_models import ChatCohere
from langchain_core.language_models.chat_models import BaseChatModel
from dotenv import load_dotenv, find_dotenv

logger = logging.getLogger(__name__)

# get env variables
load_dotenv(find_dotenv())
COHERE_API_KEY = os.environ["COHERE_API_KEY"]


def get_chat_model() -> BaseChatModel:
    """
    Returns a BaseChatModel representing the ChatCohere model loaded with the specified parameters.
    """

    logger.info(f'Loading Cohere model...')
    llm = ChatCohere(
            cache=True,
            model="command-r-plus",
            temperature=0.1,
            cohere_api_key=COHERE_API_KEY
        )
    
    return llm
