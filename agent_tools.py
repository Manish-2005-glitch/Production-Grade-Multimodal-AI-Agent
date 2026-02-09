from detector import detect
from vector_db_search import search

def detect_tool(img):
    """
    Tool: Object detection using YOLOv8
    """
    return detect(img)

def search_tool(q: str):
    """
    Tool: Semantic retrieval using FAISS
    """
    return search(q)