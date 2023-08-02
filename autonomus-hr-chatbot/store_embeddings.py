import os
import pinecone
import tiktoken
from uuid import uuid4
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings

from settings import SETTINGS
from utils import get_logger

logger = get_logger(__name__)


def store_embedding():
    """
    function to create index and store vector
    to pinecone database
    """
    # load text
    logger.info("Read text from data...")
    with open("data/hr_policy.txt", "r") as f:
        contents = f.read()
    logger.info("Succes read text...")
    tokenizer = tiktoken.get_encoding("p50k_base")

    # chunkin Function
    logger.info("Split text...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=20,
        length_function=tiktoken_len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(contents)

    # create embeddings
    logger.info("Create embedding...")
    logger.info("Use text-embedding-ada-002 from OPENAI...")
    embed = OpenAIEmbeddings(
        model='text-embedding-ada-002', 
        openai_api_key=SETTINGS["OPENAI_API_KEY"]
    )
    vectors = [(str(uuid4()), embed.embed_documents([text])[0], {"text": text}) for text in chunks]

    # create index in pinecone
    logger.info("Create index and dimension...")
    logger.info("Index_name=`tk-policy`, dimension=1536...")
    index_name = 'tk-policy'
    dimension=1536
    pinecone.init(
        api_key=SETTINGS["PINECONE_API_KEY"],
        environment=SETTINGS["PINECONE_ENV_NAME"]
    )
    # delete index if it exists
    if index_name in pinecone.list_indexes():
        pinecone.delete_index(index_name)
    # create index
    pinecone.create_index(
        name=index_name,
        metric='cosine',
        dimension=dimension       
    )

    # upsert index to pinecone
    logger.info("Upsert index to pinecone...")
    index = pinecone.Index(index_name)
    index.upsert(
        vectors=vectors,
        values=True,
        include_metadata=True
    )
    logger.info("Procces embedding succesfuly...")
    index.describe_index_stats()


def tiktoken_len(text: str):
    token = tokenizer.encode(
        text,
        disallowed_special=()
    )
    
    return len(token)

if __name__ == "__main__":
    store_embedding()




