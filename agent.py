from prompt import SYSTEM_PROMPT
from agent_tools import detect_tool, search_tool
from memory import add, get
from rag_llm import generate
from caption import caption_image
from visualizer import draw_boxes



def agent_run(query: str, image=None):
    """
    Agent execution with tool calling, RAG, Memory, and Vision Captioning.
    """

    conversation_memory = get()
    reasoning = ""

    if conversation_memory:
        reasoning += f"Conversation history:\n{conversation_memory}\n\n"

    caption = ""
    detections = ""

    
    if image is not None:
        detections = detect_tool(image)
        caption = caption_image(image)

        annotated_image = draw_boxes(image, detections)

        reasoning += f"Scene description:\n{caption}\n\n"
        reasoning += f"Detected objects:\n{detections}\n\n"


    final_prompt = f"""
    {SYSTEM_PROMPT}

    Context:
    {reasoning}

    User Question:
    {query}

    Final Answer:
    """

    answer = generate(final_prompt)

    if answer is None:
        answer = (
            f"{caption}. "
            f"The following objects were detected in the scene: {detections}."
        )

    add({
        "user": query,
        "assistant": answer
    })

    return {
    "answer": answer,
    "annotated_image": annotated_image
}

