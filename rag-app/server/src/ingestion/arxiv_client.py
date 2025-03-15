import requests
import xml.etree.ElementTree as ET
import json
import time
import dotenv
import os

dotenv.load_dotenv()

ARXIV_API_URL = os.getenv("ARXIV_API_URL")  # "http://export.arxiv.org/api/query"
DATA_PATH = os.getenv("DATA_PATH")  # './data'


def parse_arxiv_response(response: requests.Response) -> list:
    """
    Parse the arXiv response and extract the paper titles and summaries.

    Args:
        response (requests.Response): The response object from the arXiv API.

    Returns:
        list: A list of dictionaries with paper titles and summaries.
    """
    response.raise_for_status()  # Raise an error for bad responses

    # Parse the XML response
    root = ET.fromstring(response.content)
    papers = []

    # Iterate over each entry in the XML feed
    for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
        title = entry.find("{http://www.w3.org/2005/Atom}title").text
        summary = entry.find("{http://www.w3.org/2005/Atom}summary").text
        papers.append({"title": title.strip(), "summary": summary.strip()})

    return papers


def fetch_papers(query: str, max_results: int = 10) -> list:
    """
    Fetch papers from the arXiv API based on a query.

    Args:
        query (str): The search query.
        max_results (int): The maximum number of results to fetch.

    Returns:
        list: A list of dictionaries with paper titles and summaries.
    """
    params = {"search_query": query, "start": 0, "max_results": max_results}

    response = requests.get(ARXIV_API_URL, params=params)
    return parse_arxiv_response(response)
    # response.raise_for_status()  # Raise an error for bad responses

    # # Parse the XML response
    # root = ET.fromstring(response.content)
    # papers = []

    # # Iterate over each entry in the XML feed
    # for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
    #     title = entry.find('{http://www.w3.org/2005/Atom}title').text
    #     summary = entry.find('{http://www.w3.org/2005/Atom}summary').text
    #     papers.append({"title": title.strip(), "summary": summary.strip()})

    # return papers


def fetch_papers_paginated(
    query: str,
    max_results: int = 20,
    results_per_page: int = 5,
    wait_time: int = 5,
    save_local=True,
):
    start = 0
    papers = []
    for i in range(start, max_results, results_per_page):
        params = {"search_query": query, "start": i, "max_results": max_results}

        response = requests.get(ARXIV_API_URL, params=params)
        subset_papers = parse_arxiv_response(response)
        if save_local:
            with open(
                f"""{DATA_PATH}/papers_{i}_{i+results_per_page}.json""", "w"
            ) as f:
                json.dump(subset_papers, f)
        papers += subset_papers
        time.sleep(wait_time)
    return papers


if __name__ == "__main__":
    # papers = fetch_papers(query="ti:perovskite", max_results=10)
    papers = fetch_papers_paginated(
        query="ti:perovskite", max_results=20, results_per_page=5, wait_time=5
    )
    print(papers)
    # This duplicates the save_local option in the function ... TODO: tidy this up and test thoroughly.
    # with open("papers.json", "w") as f:
    #     json.dump(papers, f)
