from huggingface_hub import InferenceClient
from config import HF_API_TOKEN
import os
print("HF_API_TOKEN loaded:", bool(os.getenv("HF_API_TOKEN")))

llm = InferenceClient(
    model="HuggingFaceH4/zephyr-7b-beta",
    token=HF_API_TOKEN
)

def generate(prompt):
    """
    Generate text using HuggingFace hosted LLM.
    """
    try:
        return llm.text_generation(
            prompt,
            max_new_tokens=300,
            temperature=0.7
        )
    except Exception as e:
        print("LLM ERROR:", e)
        return "⚠️ LLM service temporarily unavailable."