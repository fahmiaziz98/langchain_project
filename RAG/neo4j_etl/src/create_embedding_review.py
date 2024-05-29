import os
import logging
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain_cohere import CohereEmbeddings


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

def setup_neo4j_vector_index(embedding) -> None:

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
        logger.info("You already put the embedding into the graph, so ignore the error...")
        logger.warning(f"Failed to set up Neo4j vector index from existing graph: {e}")

    return None

if __name__ == "__main__":
    embedding = create_embedding()
    setup_neo4j_vector_index(embedding)