from vector_db_search import search
from rag_llm import generate

def rag_answer(question: str, top_k: int=3):
    """
    Retrieval-Augmented Generation pipeline.
    """
    
    retrieved_docs = search(question, k=top_k)
    
    context = "\n".join(
        f"- {doc}" for doc in retrieved_docs
    )
    
    prompt = f"""
    You are an AI assistant answering questions using retrieved context.

    Context:
    {context}

    Question:
    {question}

    Answer using only the context above. If the context is insufficient,
    say so explicitly.
    """

    return generate(prompt)