import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

# --- 1. INITIALIZATION ---

# Load the job descriptions from our CSV file.
# 'error_bad_lines=False' skips any rows that might have formatting issues.
try:
    jobs_df = pd.read_csv('data/job_descriptions.csv', on_bad_lines='skip')
except FileNotFoundError:
    print("Error: 'data/job_descriptions.csv' not found. Make sure the file exists.")
    jobs_df = pd.DataFrame(columns=['job_id', 'title', 'description']) # Create empty df to avoid crashes

# Load a pre-trained sentence transformer model.
# 'all-MiniLM-L6-v2' is a great, fast model for semantic search.
model = SentenceTransformer('all-MiniLM-L6-v2')

# Setup ChromaDB client. We use an in-memory instance for simplicity.
client = chromadb.Client()

# --- 2. DATABASE CREATION & POPULATION ---

# Create a collection (like a table in SQL) in ChromaDB.
# We add 'if not exists' logic to prevent errors on app reload.
collection_name = "job_descriptions"
if collection_name not in [c.name for c in client.list_collections()]:
    collection = client.create_collection(name=collection_name)

    # Prepare documents and IDs for ChromaDB.
    documents = []
    metadatas = []
    ids = []
    for index, row in jobs_df.iterrows():
        # We combine title and description for richer context.
        content = f"Title: {row['title']}. Description: {row['description']}"
        documents.append(content)
        # Metadatas store extra info we want to retrieve with the match.
        metadatas.append({'title': row['title'], 'description': row['description']})
        # IDs must be unique strings.
        ids.append(str(row['job_id']))

    # Generate embeddings for all documents. This might take a moment on first run.
    embeddings = model.encode(documents).tolist()

    # Add the data to the ChromaDB collection.
    if ids: # Ensure there are documents to add
        collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
else:
    # If collection already exists, just get it.
    collection = client.get_collection(name=collection_name)


# --- 3. THE SEARCH FUNCTION ---

def find_matching_jobs(user_skills):
    """
    Finds job descriptions that best match a list of user skills.
    """
    if not user_skills:
        return []

    # Join the skills into a single string to create a "query document".
    query_text = ", ".join(user_skills)
    
    # Generate the embedding for the user's skills.
    query_embedding = model.encode(query_text).tolist()
    
    # Query the collection to find the 3 most similar job descriptions.
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3 # Ask for the top 3 matches
    )
    
    # The results contain a lot of info. We only need the 'metadatas'.
    # The results are nested, so we access the first (and only) list of matches.
    matched_jobs = results.get('metadatas', [[]])[0]
    return matched_jobs