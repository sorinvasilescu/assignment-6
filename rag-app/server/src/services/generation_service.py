import boto3
from botocore.exceptions import ClientError
import os
from typing import List, Dict, Union
from server.src.models.document import RetrievedDocument  # Import the Pydantic model
from server.src.config import Settings
from fastapi import Depends
import requests
import json
from server.src.config import settings
import opik
import openai
from openai import OpenAI

client = OpenAI()

@opik.track  # TODO: test if this works with async methods? I think it will.
def call_llm(prompt: str) -> Union[Dict, None]:
    """Call LLM API to generate a response."""
    if settings.llm_provider == "ollama":
        try:
            # Use host.docker.internal to access the host machine from inside a Docker container
            ollama_url = 'http://host.docker.internal:11434/api/generate'
            print(f"Using Ollama provider. Sending request to {ollama_url}")
            response = requests.post(
                ollama_url,
                json={
                    "model": "tinyllama",
                    "prompt": prompt,
                    "stream": False
                },
                timeout=30  # Add timeout to avoid hanging indefinitely
            )
            response.raise_for_status()
            data = response.json()
            print(f"Ollama response received successfully: {data.keys()}")
            return {
                "response": data['response'],
                "response_tokens_per_second": None  # Ollama doesn't provide this metric
            }
        except requests.exceptions.ConnectionError as e:
            print(f"Connection error to Ollama API: {e}. Is Ollama running on port 11434?")
            return None
        except requests.exceptions.Timeout as e:
            print(f"Timeout error calling Ollama API: {e}. The request took too long to complete.")
            return None
        except requests.exceptions.HTTPError as e:
            print(f"HTTP error calling Ollama API: {e}. Status code: {e.response.status_code}")
            return None
        except json.JSONDecodeError as e:
            print(f"JSON decode error from Ollama API: {e}. Response text: {response.text[:200]}")
            return None
        except KeyError as e:
            print(f"Key error with Ollama response: {e}. Response content: {data}")
            return None
        except Exception as e:
            print(f"Unexpected error calling Ollama API: {type(e).__name__}: {e}")
            return None
    else:
        try:
            response = client.chat.completions.create(
                model=settings.openai_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
                top_p=settings.top_p,
            )

            print("Successfully generated response")
            data = {"response": response.choices[0].message.content}
            data["response_tokens_per_second"] = (
                (response.usage.total_tokens / response.usage.completion_tokens)
                if hasattr(response, "usage")
                else None
            )
            print(f"call_llm returning {data}")
            print(f"data.response = {data['response']}")
            return data

        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return None  # TODO: error handling

@opik.track
async def generate_response(
    query: str,
    chunks: List[Dict],
    max_tokens: int = 200,
    temperature: float = 0.7,
) -> Dict:  # str:
    """
    Generate a response using an Ollama endpoint running locally, t
    his will be changed to allow for Bedrock later.

    Args:
        query (str): The user query.
        context (List[Dict]): The list of documents retrieved from the retrieval service.
        max_tokens (int): The maximum number of tokens to generate in the response.
        temperature (float): Sampling temperature for the model.
    """
    QUERY_PROMPT = """
    You are a helpful AI language assistant, please use the following context to answer the query. Answer in English.
    Context: {context}
    Query: {query}
    Answer:
    """
    # Concatenate documents' summaries as the context for generation
    context = "\n".join([chunk["chunk"] for chunk in chunks])
    prompt = QUERY_PROMPT.format(context=context, query=query)
    print(f"calling call_llm with provider: {settings.llm_provider} ...")
    response = call_llm(prompt)
    print(f"generate_response returning {response}")
    
    # Handle the case when call_llm returns None
    if response is None:
        if settings.llm_provider == "ollama":
            return {
                "response": "Error connecting to Ollama. Please ensure Ollama is running on your host machine at port 11434 and has the 'tinyllama' model installed. You can install it with 'ollama pull tinyllama'.",
                "response_tokens_per_second": None,
                "error": True
            }
        else:
            return {
                "response": "I apologize, but I encountered an error while generating the response. Please check your API configuration and try again.",
                "response_tokens_per_second": None,
                "error": True
            }
    
    return response  # now this is a dict.