from fastapi import APIRouter, HTTPException, Query
from typing import List
from services.retrieval_service import retrieve_top_k_chunks
from models.document import Document, RetrievedDocument
from dotenv import load_dotenv
import os
import opik

load_dotenv()

DATA_PATH = os.getenv("DATA_PATH")

# TODO: this should leverage the settings ingestor.
# Database connection configuration
db_config = {
    "dbname": os.environ.get("POSTGRES_DB"),
    "user": os.environ.get("POSTGRES_USER"),
    "password": os.environ.get("POSTGRES_PASSWORD"),
    "host": os.environ.get("POSTGRES_HOST"),
    "port": os.environ.get("POSTGRES_PORT"),
}

router = APIRouter()


@opik.track
@router.get("/retrieve", response_model=List[RetrievedDocument])
async def retrieve_top_k_chunks_endpoint(
    query: str = Query(..., description="The query text from the user"),
    top_k: int = Query(
        5, description="Number of top chunks to retrieve (default is 5)"
    ),
):
    """
    Retrieve the top K relevant chunks based on a query.

    Args:
        query (str): The query text from the user.
        top_k (int): Number of top documents to retrieve (default is 5).

    Returns:
        List[Document]: A list of the top retrieved chunks.
    """
    try:
        # TODO: Tried using await but retrieve top k is not async, can try adapting to use asyncpg in future.
        chunks = retrieve_top_k_chunks(query, top_k, db_config=db_config)
        if not chunks:
            raise HTTPException(status_code=404, detail="No chunks found.")
        return chunks
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving chunks: {str(e)}"
        )
