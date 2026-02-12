# 🚀 Production-Grade Multimodal AI Agent

End-to-end **Object Detection + Tracking + Scene Understanding + RAG + Agentic Reasoning** system built using open-source models.

A production-ready multimodal AI system capable of processing videos, tracking objects, generating captions, performing semantic search, and reasoning over visual content.

## 🏗️ Architecture

```

Video Input

    ↓

YOLOv8 Detection

    ↓

DeepSORT Tracking

    ↓

Frame Captioning

    ↓

Embedding Generation

    ↓

FAISS Vector Store

    ↓

RAG + Agent Reasoning

    ↓

API Response / UI Output

```

---

## 📁 Project Structure

```

.

├── app.py                  # FastAPI entry point

├── video_tracker.py        # Detection + Tracking pipeline

├── caption.py              # Image captioning

├── rag_llm.py              # RAG logic

├── vector_db_store.py      # FAISS storage

├── vector_db_search.py     # FAISS search

├── agent.py                # Agent reasoning logic

├── frontend.py             # Streamlit UI

├── requirements.txt

├── Dockerfile

└── .dockerignore

```

# 🐳 Docker Deployment

### Build Image

```bash

docker build -t multimodal-ai-agent .

```

### Run Container

```bash

docker run -p 8000:8000 multimodal-ai-agent

```

---

# 📊 Tech Stack

- Python 3.10  

- FastAPI  

- Streamlit  

- YOLOv8 (Ultralytics)  

- DeepSORT  

- Transformers  

- Sentence-Transformers  

- FAISS  

- LangChain  

- Docker  

---

# 🔥 Example Use Cases

- Smart surveillance systems  

- AI-powered video search engines  

- Scene summarization systems  

- Multimodal research projects  

- Video analytics platforms  

---

# 👨‍💻 Author

**Manish Mohapatra**  

GitHub: https://github.com/Manish-2005-glitch  

---

# ⭐ If You Like This Project

Give it a star ⭐  

It motivates further open-source development!

