import streamlit as st
import requests
import os
import tempfile
import base64
import numpy as np
import cv2

BACKEND_API_URL = (
    #"BACKEND_API_URL",
    "http://localhost:8000/agent"
)

st.set_page_config(
    page_title="VisionRAG Agent",
    layout="wide"
)

st.title("🧠 VisionRAG Agent")
st.caption(
    "YOLOv8 • DeepSORT • BLIP • FAISS • RAG • Agentic AI"
)

st.sidebar.header("Inputs")

uploaded_file = st.sidebar.file_uploader(
    "Upload an image",
    type=["png", "jpg", "jpeg"]
)

question = st.sidebar.text_input(
    "Ask a question",
    value="Describe what is happening in this scene"
)

run_button = st.sidebar.button("Run Agent")

if run_button:
    if not question:
        st.warning("Please enter a question.")
    else:
        st.info("⏳ Running agent...")

        files = None
        tmp_path = None

        if uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            files = {"file": open(tmp_path, "rb")}

        try:
            response = requests.post(
                BACKEND_API_URL,
                data={"question": question},
                files=files,
                timeout=120
            )

            if response.status_code == 200:
                result = response.json()

                st.subheader("🧠 Agent Response")
                st.success(result["response"])

                if "image" in result:
                    img_bytes = base64.b64decode(result["image"])
                    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                    st.subheader("📦 Detected Objects (Bounding Boxes)")
                    st.image(img[:, :, ::-1], width=900)

                elif tmp_path:
                    st.subheader("📷 Uploaded Image")
                    st.image(tmp_path, width=900)

            else:
                st.error(f"Backend error ({response.status_code}).")

        except Exception as e:
            st.error(f"❌ Failed to reach backend: {e}")
