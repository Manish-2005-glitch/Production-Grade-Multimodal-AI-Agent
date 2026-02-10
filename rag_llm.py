from huggingface_hub import InferenceClient
from config import HF_API_TOKEN

# Primary (cloud)
hf_llm = InferenceClient(
    model="google/flan-t5-base",
    token=HF_API_TOKEN,
    timeout=30
)

def generate(prompt: str):
    try:
        return hf_llm.text_generation(
            prompt,
            max_new_tokens=200,
            temperature=0.7
        )
    except Exception as e:
        print("LLM ERROR:", e)
        return None


