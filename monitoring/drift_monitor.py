import numpy as np
import ast
from database.supabase_client import supabase


def fetch_embeddings():

    response = supabase.table("interaction_log").select("embedding").execute()

    embeddings = [ast.literal_eval(row["embedding"]) for row in response.data]

    return np.array(embeddings)


def detect_embedding_drift():

    embeddings = fetch_embeddings()

    if len(embeddings) < 10:
        print("Not enough data for drift detection")
        return

    midpoint = len(embeddings) // 2

    old_embeddings = embeddings[:midpoint]
    new_embeddings = embeddings[midpoint:]

    old_centroid = np.mean(old_embeddings, axis=0)
    new_centroid = np.mean(new_embeddings, axis=0)

    drift_score = np.linalg.norm(old_centroid - new_centroid)

    print("Drift score:", drift_score)

    if drift_score > 0.35:
        print("⚠ Drift detected!")
    else:
        print("System stable")