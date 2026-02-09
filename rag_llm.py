from huggingface_hub import InferenceClient
from config import HF_API_TOKEN

llm = InferenceClient(
    model="deepseek-ai/DeepSeek-V3.2",
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
        return "LLM service temporarily unavailable."