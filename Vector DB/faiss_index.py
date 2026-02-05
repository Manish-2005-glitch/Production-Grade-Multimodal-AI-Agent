import faiss

DIM = 384

def create():
    return faiss.IndexFlatL2(DIM)