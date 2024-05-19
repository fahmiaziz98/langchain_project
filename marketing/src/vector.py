import pandas as pd
import streamlit as st
from langchain.document_loaders import DataFrameLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_cohere.embeddings import CohereEmbeddings
from langchain.vectorstores import VectorStore

# Set Env
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
COHERE_API_KEY = os.environ["COHERE_API_KEY"]


def vector_database(data, col1_name, col2_name) -> VectorStore:
    embeddings = CohereEmbeddings(
            cohere_api_key=COHERE_API_KEY,
    )
    text_splitter = CharacterTextSplitter(
            chunk_size=512, chunk_overlap=50
    )

    db_path = f"./tempfolder/db_{os.path.basename(data).split('.')[0]}"
    if os.path.exists(db_path):
        print("Using Cached One")
        data = pd.read_csv(data, usecols=[col1_name, col2_name])
        st.write("Click the row for details. Sample Data:")
        st.write(data.sample(5))

        db = Chroma(embedding_function=embeddings, persist_directory=db_path)
    else:
        data = pd.read_csv(data, usecols=[col1_name, col2_name])
        
        st.write("Click the row for details. Sample Data:")
        st.write(data.sample(5))

        data["summary"] = data[col1_name] + data[col2_name]
        loader = DataFrameLoader(data, page_content_column="summary")
        documents = loader.load()
        documents.extend(loader.load_and_split())
        
        docs = text_splitter.split_documents(documents)

        db = Chroma.from_documents(
            docs, 
            embeddings,
            persist_directory=db_path
        )
    
    return db
