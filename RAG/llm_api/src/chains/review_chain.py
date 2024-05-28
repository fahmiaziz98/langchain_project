import os
import logging
import time
from dotenv import load_dotenv, find_dotenv
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain_cohere import CohereEmbeddings, ChatCohere
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

load_dotenv(find_dotenv())
HOSPITAL_QA_MODEL = os.getenv("HOSPITAL_QA_MODEL")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

def create_embedding():
    logger.info("Load CohereEmbedding...")
    return CohereEmbeddings(cohere_api_key=COHERE_API_KEY)

def setup_neo4j_vector_index(embedding):
    neo4j_vector_index = None

    logger.info("Attempting to set up Neo4j vector index from existing graph...")
    try:
        neo4j_vector_index = Neo4jVector.from_existing_graph(
            embedding=embedding,
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            index_name="reviews",
            node_label="Review",
            text_node_properties=[
                "physician_name",
                "patient_name",
                "text",
                "hospital_name",
            ],
            embedding_node_property="embedding",
        )
    except Exception as e:
        logger.warning(f"Failed to set up Neo4j vector index from existing graph: {e}")

    if not neo4j_vector_index:
        logger.info("Attempting to set up Neo4j vector index from existing index...")
        try:
            neo4j_vector_index = Neo4jVector.from_existing_index(
                embedding=embedding,
                url=NEO4J_URI,
                username=NEO4J_USERNAME,
                password=NEO4J_PASSWORD,
                index_name="reviews",
                node_label="Review",
                embedding_node_property="embedding",
            )
        except Exception as e:
            logger.error(f"Failed to set up Neo4j vector index from existing index: {e}")

    if not neo4j_vector_index:
        raise RuntimeError("Failed to set up Neo4j vector index")

    return neo4j_vector_index

def setup_vector_chain():
    logger.info("Setting up vector chain...")
    embedding = create_embedding()

    neo4j_vector_index = setup_neo4j_vector_index(embedding)

    review_template = """
    Your job is to use patient reviews to answer questions about their experience at a hospital. 
    Use the following context to answer questions. Be as detailed as possible, but don't make up any information that's not from the context. 
    If you don't know an answer, say you don't know.
    {context}
    """

    system_prompt = SystemMessagePromptTemplate(
        prompt=PromptTemplate(
            input_variables=["context"],
            template=review_template
        )
    )

    human_prompt = HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            input_variables=["question"],
            template="{question}"
        )
    )

    messages = [system_prompt, human_prompt]
    review_prompt = ChatPromptTemplate(
        input_variables=["context", "question"],
        messages=messages
    )

    vector_chain = RetrievalQA.from_chain_type(
        llm=ChatCohere(
            model=HOSPITAL_QA_MODEL,
            cohere_api_key=COHERE_API_KEY,
            temperature=0.1
        ),
        chain_type="stuff",
        retriever=neo4j_vector_index.as_retriever(k=10),
    )

    vector_chain.combine_documents_chain.llm_chain.prompt = review_prompt

    return vector_chain

def run_query(vector_chain, query):
    logger.info("Starting query...")
    start_time = time.time()
    response = vector_chain.invoke(query)
    end_time = time.time()
    response_time = end_time - start_time
    logger.info(f"Response time: {response_time} seconds")

    if response and "result" in response:
        return response.get("result")
    else:
        logger.warning("No result found in response")
        return None

# if __name__ == "__main__":
#     logger.info("Setting up...")
#     try:
#         vector_chain = setup_vector_chain()
#         query = """What have patients said about hospital efficiency? Mention details from specific reviews."""
#         response = run_query(vector_chain, query)
#         if response:
#             print(response)
#         else:
#             print("No response received.")
#     except Exception as e:
#         logger.error(f"Error in main execution: {e}")
