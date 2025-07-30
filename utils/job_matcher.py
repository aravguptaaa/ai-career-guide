import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

try:
    jobs_df = pd.read_csv('data/job_descriptions.csv', on_bad_lines='skip')
except FileNotFoundError:
    print("Error: 'data/job_descriptions.csv' not found. Make sure the file exists.")
    jobs_df = pd.DataFrame(columns=['job_id', 'title', 'description'])

model = SentenceTransformer('all-MiniLM-L6-v2')

client = chromadb.Client()

collection_name = "job_descriptions"
if collection_name not in [c.name for c in client.list_collections()]:
    collection = client.create_collection(name=collection_name)

    documents = []
    metadatas = []
    ids = []
    for index, row in jobs_df.iterrows():

        content = f"Title: {row['title']}. Description: {row['description']}"
        documents.append(content)
        metadatas.append({'title': row['title'], 'description': row['description']})
        ids.append(str(row['job_id']))

    embeddings = model.encode(documents).tolist()

    if ids:
        collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
else:
    collection = client.get_collection(name=collection_name)


def find_matching_jobs(user_skills):
    """
    Finds job descriptions that best match a list of user skills.
    """
    if not user_skills:
        return []


    query_text = ", ".join(user_skills)
    
    query_embedding = model.encode(query_text).tolist()
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3 # Ask for the top 3 matches
    )
    
    # The results contain a lot of info. We only need the 'metadatas'.
    # The results are nested, so we access the first (and only) list of matches.
    matched_jobs = results.get('metadatas', [[]])[0]
    return matched_jobs
