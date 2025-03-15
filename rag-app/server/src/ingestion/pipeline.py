from arxiv_client import fetch_papers
from embeddings import chunk_text, generate_embeddings, process_papers
from utils import read_json_files
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv
import os

load_dotenv()

DATA_PATH = os.getenv('DATA_PATH')

# Database connection configuration
db_config = {
    "dbname": os.environ.get("POSTGRES_DB"), 
    "user": os.environ.get("POSTGRES_USER"),
    "password": os.environ.get("POSTGRES_PASSWORD"),
    "host": os.environ.get("POSTGRES_HOST"),
    "port": os.environ.get("POSTGRES_PORT")
}

def insert_papers_to_pgvector(data: list, db_config: dict):
    """
    Inserts a list of papers into the Postgres table with pgvector.
    
    Args:
        data (list): A list of dictionaries, where each dict contains title, summary, chunks, and embeddings.
        db_config (dict): Dictionary containing Postgres connection details (dbname, user, password, host, port).
    
    """
    # Establish the database connection
    conn = psycopg2.connect(**db_config)
    cursor = conn.cursor()

    # SQL query to insert a paper's title, summary, chunk, and embedding into the 'papers' table
    insert_query = """
    INSERT INTO papers (title, summary, chunk, embedding)
    VALUES %s
    """
    
    # Prepare the data for insertion
    values = []
    for entry in data:
        title = entry["title"]
        summary = entry["summary"]
        chunks = entry["chunks"]
        embeddings = entry["embeddings"]

        # Ensure chunks and embeddings are the same length
        assert len(chunks) == len(embeddings), "Mismatch between chunks and embeddings length."
        
        # For each chunk and its corresponding embedding, prepare a row for insertion
        for chunk, embedding in zip(chunks, embeddings):
            # Prepare each value (embedding should be converted to a list or array-like format for insertion)
            values.append((title, summary, chunk, embedding.tolist()))  # .tolist() converts numpy array to list
    
    # Use psycopg2's execute_values for efficient bulk insertion
    execute_values(cursor, insert_query, values)
    
    # Commit the transaction and close the connection
    conn.commit()
    cursor.close()
    conn.close()

    print(f"Successfully inserted {len(values)} rows into the papers table.")
# def save_to_db(papers, embeddings):
#     """
#     Save the paper titles, summaries, and embeddings to Postgres.
    
#     Args:
#         papers (list): A list of papers with titles and summaries.
#         embeddings (list): A list of embeddings for the corresponding papers.
#     """
#     conn = psycopg2.connect(**db_config)
#     cursor = conn.cursor()

#     insert_query = """
#     INSERT INTO papers (title, summary, embedding) 
#     VALUES %s
#     """
    
#     # Prepare values for bulk insertion
#     values = [
#         (paper['title'], paper['summary'], embedding.tolist())
#         for paper, embedding in zip(papers, embeddings)
#     ]
    
#     # Bulk insert
#     execute_values(cursor, insert_query, values)
#     conn.commit()
#     cursor.close()
#     conn.close()

    
def run_pipeline(json_dir: str, chunk_size: int = 512, overlap: int = 50):
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
    #return processed_papers
    # Step 3: Save the processed papers with embeddings to pgvector
    try:
        insert_papers_to_pgvector(processed_papers, db_config)
    except Exception as e:
        print(f"Error: {e}")
    
    #save_processed_papers_to_file(processed_papers, output_file)
    #print(processed_papers[0])
    #print(f"Successfully processed and saved {len(processed_papers)} papers.")


if __name__ == "__main__":
    # Example: Run the pipeline with a query
    print(f'''Reading JSON files from {DATA_PATH}...''')
    run_pipeline(json_dir=DATA_PATH)
    print(f'''Completed ingestion into database {db_config['dbname']}''')