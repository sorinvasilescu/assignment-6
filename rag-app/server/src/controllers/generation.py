from fastapi import APIRouter, HTTPException, Query
from typing import List
from services.generation_service import generate_response
from services.retrieval_service import retrieve_top_k_chunks
from services.query_expansion_service import expand_query
from models.document import RetrievedDocument
import os
from config import settings
import opik

router = APIRouter()

# Reuse your database configuration
db_config = {
    "dbname": os.environ.get("POSTGRES_DB"),
    "user": os.environ.get("POSTGRES_USER"),
    "password": os.environ.get("POSTGRES_PASSWORD"),
    "host": os.environ.get("POSTGRES_HOST"),
    "port": os.environ.get("POSTGRES_PORT"),
}

# TODO: move all this model config to config files!
@opik.track
@router.get("/generate")
async def generate_answer_endpoint(
    query: str = Query(..., description="The query text from the user"),
    top_k: int = Query(5, description="Number of top chunks to retrieve"),
    max_tokens: int = Query(
        200, description="The maximum number of tokens to generate"
    ),
    temperature: float = Query(0.7, description="Sampling temperature for the model"),
):
    """
    Retrieve the top K relevant chunks and generate a response based on them.

    Args:
        query (str): The query text from the user.
        top_k (int): Number of top chunks to retrieve.
        max_tokens (int): Maximum number of tokens to generate in the response.
        temperature (float): Temperature setting for the generation model.

    Returns:
        str: The generated answer based on the query and retrieved chunks.
    """

    try:
        query_expanded = False
        # query = expand_query(query)
        # print(f"Expanded query is: {query} and of type {type(query)}")
        # if not query:
        #     raise HTTPException(status_code=400, detail="Query expansion failed.")
        # else:
        #     query_expanded = True

        # Retrieve documents
        chunks = retrieve_top_k_chunks(query, top_k, db_config=db_config)
        if not chunks:
            raise HTTPException(status_code=404, detail="No documents found.")

        # Pass the RetrievedDocument objects directly
        generated_response = await generate_response(
            query, chunks, max_tokens, temperature
        )  # is this sync?
        generated_response["query_expanded"] = query_expanded
        print(f"Generated response {generated_response}")
        return generated_response  # {"response": generated_response}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating response: {str(e)}"
        )
