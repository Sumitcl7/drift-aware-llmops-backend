import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import ast
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from database.supabase_client import supabase


def fetch_embeddings():

    response = supabase.table("interaction_log").select("embedding").execute()

    embeddings = [ast.literal_eval(row["embedding"]) for row in response.data]

    return np.array(embeddings)


embeddings = fetch_embeddings()

midpoint = len(embeddings) // 2

old_embeddings = embeddings[:midpoint]
new_embeddings = embeddings[midpoint:]

combined = np.vstack((old_embeddings, new_embeddings))

pca = PCA(n_components=2)

reduced = pca.fit_transform(combined)

old_points = reduced[:midpoint]
new_points = reduced[midpoint:]

plt.scatter(old_points[:,0], old_points[:,1], label="Old Queries", alpha=0.7)
plt.scatter(new_points[:,0], new_points[:,1], label="New Queries", alpha=0.7)

plt.title("Embedding Drift Visualization")
plt.legend()

plt.show()