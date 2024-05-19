from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import Chroma

import os
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv, find_dotenv

# get env variables
load_dotenv(find_dotenv())
COHERE_API_KEY = os.environ["COHERE_API_KEY"]


def embedding_store(doc):
    """
    A function that stores embeddings and retrieves a document using Chroma based on the provided document path.

    Parameters:
        doc (str): The path to the document for which embeddings are to be stored.

    Returns:
        retriever: The retriever object for the stored embeddings and document.
    """
    # Set embeddings
    embeddings = CohereEmbeddings(cohere_api_key=COHERE_API_KEY)
    # Split
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=512, chunk_overlap=50, separators=[" ", ",", "\n"]
    )

    db_path = f"./tempfolder/db_{os.path.basename(doc).split('.')[0]}"
    if os.path.exists(db_path):
        print("Using Cached One")
        db_chroma = Chroma(embedding_function=embeddings, persist_directory=db_path)
        retriever = db_chroma.as_retriever()
    else:
        loader = PyPDFLoader(doc)
        pages = loader.load_and_split()
        docs = text_splitter.split_documents(pages)
        db_chroma = Chroma.from_documents(
            docs,
            embeddings,
            persist_directory=db_path,
        )
        retriever = db_chroma.as_retriever()


    return retriever


