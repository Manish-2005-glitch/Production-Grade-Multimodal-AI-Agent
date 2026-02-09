import pickle
import os
import numpy as np
from vector_db_store import embed

DB_PATH = "db.pkl"

def search(query, k=3):
    if not os.path.exists(DB_PATH):
        return []

    index, meta = pickle.load(open(DB_PATH, "rb"))

    q_vec = np.array([embed(query)], dtype=np.float32)
    D, I = index.search(q_vec, k)

    return [meta[i] for i in I[0]]
