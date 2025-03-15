from sentence_transformers import SentenceTransformer
from typing import List
import os
import json
from utils import read_json_files, save_processed_papers_to_file
import dotenv

dotenv.load_dotenv()

DATA_PATH = os.getenv('DATA_PATH')

# Load a pre-trained Sentence Transformer model (e.g., 'all-MiniLM-L6-v2')
model = SentenceTransformer('all-MiniLM-L6-v2') # TODO: Replace with Bedrock embeddings.

def chunk_text(text: str, max_length: int = 512, overlap: int = 50) -> List[str]:
    """
    Chunk the text into smaller pieces with some overlap between chunks.
    
    Args:
        text (str): The text to chunk.
        max_length (int): The maximum number of tokens in each chunk.
        overlap (int): The number of overlapping tokens between adjacent chunks.
        
    Returns:
        List[str]: A list of text chunks with the specified overlap.
    """
    words = text.split()
    chunks = []

    # Ensure the overlap is smaller than max_length
    if overlap >= max_length:
        raise ValueError("Overlap must be smaller than the maximum chunk length.")
    
    # Slide through the text with a window that overlaps
    start = 0
    while start < len(words):
        end = min(start + max_length, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        # Move the window by max_length - overlap to create overlap between chunks
        start += max_length - overlap

    return chunks

def generate_embeddings(text_chunks: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a list of text chunks.
    
    Args:
        text_chunks (List[str]): The list of text chunks.
        
    Returns:
        List[List[float]]: A list of embeddings (one embedding per chunk).
    """
    embeddings = model.encode(text_chunks, convert_to_tensor=False)
    return embeddings

def process_papers(papers: List[dict], chunk_size: int = 512, overlap: int = 50):
    """
    Process a list of papers: chunk their summaries and generate embeddings.
    
    Args:
        papers (List[dict]): List of papers with title and summary.
        chunk_size (int): Maximum number of tokens per chunk.
        overlap (int): Number of overlapping tokens between chunks.
        
    Returns:
        List[dict]: A list of processed papers with embeddings.
    """
    processed_papers = []
    
    for paper in papers:
        title = paper.get("title")
        summary = paper.get("summary")

        # Chunk the summary into smaller chunks
        chunks = chunk_text(summary, max_length=chunk_size, overlap=overlap)
        
        # Generate embeddings for each chunk
        embeddings = generate_embeddings(chunks)
        
        processed_papers.append({
            "title": title,
            "summary": summary,
            "chunks": chunks,
            "embeddings": embeddings
        })
    
    return processed_papers




def run_pipeline(json_dir: str, output_file: str, chunk_size: int = 512, overlap: int = 50):
    """
    Run the complete pipeline: read JSON files, process papers, and save results.
    
    Args:
        json_dir (str): Directory containing JSON files with papers.
        output_file (str): Path to save the output JSON file.
        chunk_size (int): The maximum number of tokens per chunk.
        overlap (int): Number of overlapping tokens between chunks.
    """
    # Step 1: Read JSON files
    papers = read_json_files(json_dir)
    
    # Step 2: Process papers (chunking and embedding)
    processed_papers = process_papers(papers, chunk_size=chunk_size, overlap=overlap)
    print(f"Succesfully processed {len(processed_papers)} papers.")
    return processed_papers
    # Step 3: Save the processed papers with embeddings
    #save_processed_papers_to_file(processed_papers, output_file)
    #print(processed_papers[0])
    #print(f"Successfully processed and saved {len(processed_papers)} papers.")

if __name__ == "__main__":
    # Example usage: process papers in the "data/json_files" directory and save to "output/processed_papers.json"
    run_pipeline(json_dir=DATA_PATH, output_file=f'''{DATA_PATH}/processed_papers.json''')