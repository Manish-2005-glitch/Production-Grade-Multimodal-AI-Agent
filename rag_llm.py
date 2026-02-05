from huggingface_hub import InferenceClient
from config import HF_API_TOKEN

llm = InferenceClient(
    model="deepseek-ai/DeepSeek-V3.2",
    token=HF_API_TOKEN
)

def generate(prompt):
    return llm.text_generation(prompt, max_new_tokens=300)
