import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from faiss_index import create
from dotenv import load_dotenv

load_dotenv()

model = SentenceTransformer("all-MiniLM-L6-v2")

def embed(text: str):
    """
    Converts text into a normalized vector embedding.
    """
    return model.encode(text, normalize_embeddings=True)

def store(texts):
    index = create()
    meta = []
    for t in texts:
        index.add(np.array([embed(t)]))
        meta.append(t)
        
    pickle.dump((meta, index), open("db.pkl", "wb"))