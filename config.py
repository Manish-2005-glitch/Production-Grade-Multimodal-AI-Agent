from dotenv import load_dotenv
import os

load_dotenv()

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
BACKEND_API_URL = os.getenv("BACKEND_API_URL")