"""Retrieve documents service

This will perform naive rag retrieval for a given query using cosine similarity and top_k retrieval
"""
import psycopg2
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import opik

# Load a pre-trained Sentence Transformer model (e.g., 'all-MiniLM-L6-v2') - ideally retrieve this from app state ...
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def get_db_connection(db_config: dict):
    """
    Establishes a connection to the Postgres database.

    Args:
        db_config (dict): Dictionary containing Postgres connection details (dbname, user, password, host, port).

    Returns:
        psycopg2.connection: The connection object.
    """
    return psycopg2.connect(**db_config)


@opik.track
def retrieve_top_k_chunks(query: str, top_k: int, db_config: dict) -> List[Dict]:
    """
    Retrieves the top_k documents based on cosine similarity to the query embedding using pgvector.

    Args:
        query (str): The input query.
        top_k (int): The number of top chunks to retrieve.
        db_config (dict): Dictionary containing Postgres connection details.

    Returns:
        List[Dict]: A list of dictionaries containing the top_k chunks with their titles and summaries.
    """
    # Generate the embedding for the query
    # embedding_model = (
    #     app.state.embedding_model
    # )  # TODO: get reference to app state from Request...
    query_embedding = embedding_model.encode(
        query, convert_to_tensor=False
    ).tolist()  # Need list converstion for pgvector to interpret correctly

    # Connect to the database
    conn = get_db_connection(db_config)

    try:
        cursor = conn.cursor()

        # SQL query to find the top_k chunks using cosine similarity
        query = """
        SELECT id, title, chunk, embedding <=> %s::vector AS similarity
        FROM papers
        ORDER BY similarity ASC
        LIMIT %s;
        """

        # Execute the query with the query embedding and top_k value
        cursor.execute(query, (query_embedding, top_k))
        rows = cursor.fetchall()

        # Prepare the results
        results = [
            {"id": row[0], "title": row[1], "chunk": row[2], "similarity_score": row[3]}
            for row in rows
        ]

        return results

    finally:
        cursor.close()
        conn.close()
