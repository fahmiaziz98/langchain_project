import os
import logging
import time
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain_cohere import CohereEmbeddings, ChatCohere
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from dotenv import load_dotenv, find_dotenv
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
    """Load the CohereEmbeddings model for embeddings.

    This function loads the CohereEmbeddings model for use in the review_chain.
    The COHERE_API_KEY environment variable must be set for this function to
    work.

    Returns:
        CohereEmbeddings: The loaded CohereEmbeddings model.
    """
    logger.info("Load CohereEmbedding...")
    return CohereEmbeddings(cohere_api_key=COHERE_API_KEY)


def setup_neo4j_vector_index(embedding: CohereEmbeddings) -> Neo4jVector:
    """
    Attempt to set up a Neo4j vector index using an existing index.

    This function will attempt to use an existing index in a Neo4j database
    to store the embeddings from the given CohereEmbeddings model. If the index
    already exists, it will be used; otherwise, an error will be raised.

    Args:
        embedding: The CohereEmbeddings model from which to load the embeddings.

    Returns:
        Neo4jVector: The loaded Neo4j vector index.

    Raises:
        RuntimeError: If the index cannot be found and set up.
    """
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
    """
    Set up a vector chain for querying patient reviews.

    This function creates a vector chain that is capable of answering questions
    about patient reviews using a Neo4j vector index and a Cohere chat model.

    Returns:
        A vector chain that can be used to answer questions about patient reviews.
    """
    logger.info("Setting up vector chain...")
    embedding = create_embedding()
    neo4j_vector_index = setup_neo4j_vector_index(embedding)

    review_template = """
    Your job is to use patient reviews to answer questions about their experience at a hospital. 
    Use the following context to answer questions. Be as detailed as possible, but don't make up any information that's not from the context. 
    If you don't know an answer, say you don't know.
    {context}
    """

    # System prompt - asks the user for a context and a question
    system_prompt = SystemMessagePromptTemplate(
        prompt=PromptTemplate(
            input_variables=["context"],
            template=review_template
        )
    )

    # Human prompt - asks the user for a question
    human_prompt = HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            input_variables=["question"],
            template="{question}"
        )
    )

    # Combine the system and human prompts into a single chat prompt
    messages = [system_prompt, human_prompt]
    review_prompt = ChatPromptTemplate(
        input_variables=["context", "question"],
        messages=messages
    )

    # Create a retrieval QA model with the Neo4j vector index and a Cohere chat model
    vector_chain = RetrievalQA.from_chain_type(
        llm=ChatCohere(
            model=HOSPITAL_QA_MODEL,
            cohere_api_key=COHERE_API_KEY,
            temperature=0.1
        ),
        chain_type="stuff",
        retriever=neo4j_vector_index.as_retriever(k=10),
    )

    # Set the review prompt for the chat model in the vector chain
    vector_chain.combine_documents_chain.llm_chain.prompt = review_prompt

    return vector_chain
