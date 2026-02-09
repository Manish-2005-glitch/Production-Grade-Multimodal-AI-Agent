from prompt import SYSTEM_PROMPT
from agent_tools import detect_tool, search_tool
from memory import add, get
from rag_llm import generate

def agent_run(query:str, image=None):
    """
    Agent execution with tool calling, RAG and Memory.
    """
    
    conversation_memory = get()
    
    reasoning = ""
    
    if conversation_memory:
        reasoning += f"conversation history:\n{conversation_memory}\n\n"
        
    if image is not None:
        detections = detect_tool(image)
        reasoning += f"Detected objects:\n{detections}\n\n"
        
    retrieved = search_tool(query)
    reasoning += f"Retrieved knowledge:\n{retrieved}\n\n"
    
    final_prompt = f"""
    {SYSTEM_PROMPT}

    Context:
    {reasoning}

    User Question:
    {query}

    Final Answer:
    """
    
    answer = generate(final_prompt)
    
    add({
        "user": query,
        "assistant": answer
    })

    return answer