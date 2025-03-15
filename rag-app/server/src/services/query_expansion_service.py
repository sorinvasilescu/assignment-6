from server.src.services.generation_service import generate_response, call_llm
from typing import Dict, Union
import opik


@opik.track
def expand_query(query: str) -> Union[str, None]:
    """
    Expands the query by generating a response using the OpenAI API.

    Args:
        query (str): The input query to be expanded.

    Returns:
        str: The expanded query.
    """
    expansion_prompt = f"""
    Expand the following query, specifically add relevant synoyms for key topics and phrases, do this such that
    you increase the chances of a relevant retrieval from the knowledge base. Consider that the database contains
    data pertaining to the topic in hand but it may not be extensive. The expanded query that you return should
    include the original query I provide to you below as well.

    Query: {query}
    """
    return call_llm(expansion_prompt)["response"].replace('"', "")
