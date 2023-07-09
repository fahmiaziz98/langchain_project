import os
import pinecone
import tiktoken
from uuid import uuid4
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings

from settings import SETTINGS

# load text
with open("data/hr_policy.txt", "r") as f:
    contents = f.read()

tokenizer = tiktoken.get_encoding("p50k_base")

def tiktoken_len(text: str):
    token = tokenizer.encode(
        text,
        disallowed_special=()
    )
    
    return len(token)

# chunkin Function
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=20,
    length_function=tiktoken_len,
    separators=["\n\n", "\n", " ", ""]
)

chunks = text_splitter.split_text(contents)

# create embeddings
embed = OpenAIEmbeddings(
    model='text-embedding-ada-002', 
    openai_api_key=SETTINGS["OPENAI_API_KEY"]
)
vectors = [(str(uuid4()), embed.embed_documents([text])[0], {"text": text}) for text in chunks]

# create index in pinecone
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
index = pinecone.Index(index_name)
index.upsert(
    vectors=vectors,
    values=True,
    include_metadata=True
)
index.describe_index_stats()