import pickle, numpy as np
from vector_db_store import embed

def search(q, k=3):
    index, meta = pickle.load(open("db.pkl", "rb"))
    D, I = index.search(np.array([embed(q)]), k)
    return [meta[i] for i in I[0]]