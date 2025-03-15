from typing import List
import os 
import json

def read_json_files(directory: str) -> List[dict]:
    """
    Read and load JSON files from a directory, each containing a list of papers.
    
    Args:
        directory (str): The directory containing JSON files.
        
    Returns:
        List[dict]: A list of dictionaries containing paper titles and summaries.
    """
    all_data = []
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            filepath = os.path.join(directory, filename)
            with open(filepath, "r") as f:
                papers = json.load(f)
                all_data.extend(papers)  # Append to the list of all data
    return all_data

def save_processed_papers_to_file(processed_papers: List[dict], output_file: str):
    """
    Save processed papers (with embeddings) to a JSON file.
    
    Args:
        processed_papers (List[dict]): The processed papers with embeddings.
        output_file (str): The file path to save the output JSON.
    """
    with open(output_file, "w") as f:
        json.dump(processed_papers, f, indent=4)
