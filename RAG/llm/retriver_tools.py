import os
from dotenv import load_dotenv, find_dotenv
import logging

from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_cohere import CohereEmbeddings

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def load_environment():
    load_dotenv(find_dotenv())
    cohere_api_key = os.environ["COHERE_API_KEY"]
    chroma_path = "db_vector"
    return cohere_api_key, chroma_path

def load_split_documents():
    spliter = RecursiveCharacterTextSplitter()

    loader = CSVLoader("data/reviews_20.csv", source_column="review")
    documents = loader.load()
    documents.extend(loader.load_and_split())

    docs = spliter.split_documents(documents)
    return docs

def load_to_chroma(docs, cohere_api_key, chroma_path):
    db = Chroma.from_documents(
        docs,
        CohereEmbeddings(cohere_api_key=cohere_api_key),
        persist_directory=chroma_path,
    )
    return db

def main():
    logger.info("Start...")
    cohere_api_key, chroma_path = load_environment()

    logger.info("Starting load to chroma DB process")

    logger.info("Loading dataset")
    docs = load_split_documents()

    logger.info("Loading to chroma DB")
    load_to_chroma(docs, cohere_api_key, chroma_path)

    logger.info("Finished loading to chroma DB")

if __name__ == "__main__":
    main()
