import io
import os
import pickle
from pathlib import Path

import numpy as np
import torch
import streamlit as st
from PIL import Image
from rembg import remove
from torchvision import transforms

# ── Config ───────────────────────────────────────────────
INDEX_FILE = "/app/index/jewelry_index.pkl"
BACKUP_DIR = "/app/backups"
ERROR_LOG  = "/app/index/errors.txt"
TOP_K      = 10

st.set_page_config(page_title="Jewelry Search", layout="wide")
st.title("💎 Jewelry Visual Search")

# ── Load model (cached, loads once) ──────────────────────
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
    model.eval().to(device)
    return model, device

# ── Load index (cached) ───────────────────────────────────
@st.cache_data
def load_index():
    if os.path.exists(INDEX_FILE):
        with open(INDEX_FILE, "rb") as f:
            return pickle.load(f)
    # fallback to latest backup
    backups = sorted(Path(BACKUP_DIR).glob("*.pkl"))
    if backups:
        st.warning("Main index missing — loaded from latest backup.")
        with open(backups[-1], "rb") as f:
            return pickle.load(f)
    return {}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ── Embedding ─────────────────────────────────────────────
def get_embedding(image_bytes, model, device):
    cleaned = remove(image_bytes)
    img     = Image.open(io.BytesIO(cleaned)).convert("RGB")
    tensor  = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(tensor).squeeze().cpu().numpy()
    return emb

# ── Search ────────────────────────────────────────────────
def search(query_emb, index, top_k=TOP_K):
    keys   = list(index.keys())
    matrix = np.stack([index[k]["embedding"] for k in keys])
    q_norm = query_emb / np.linalg.norm(query_emb)
    m_norm = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    scores = m_norm @ q_norm
    top_i  = np.argsort(scores)[::-1][:top_k]
    return [(index[keys[i]]["path"], float(scores[i])) for i in top_i]

# ── Load ──────────────────────────────────────────────────
model, device = load_model()
index         = load_index()

st.write(f"Inventory loaded: **{len(index)} images**")

if len(index) == 0:
    st.warning("Index is empty. Run the ingestion first (see sidebar).")

# ── Upload & Search ───────────────────────────────────────
uploaded = st.file_uploader(
    "Upload client images (max 5)",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True
)

if uploaded:
    if len(uploaded) > 5:
        st.error("Please upload a maximum of 5 images at a time.")
    elif len(index) == 0:
        st.error("Index is empty. Please run ingestion first.")
    else:
        for up in uploaded:
            st.subheader(f"Results for: {up.name}")
            col_q, col_r = st.columns([1, 4])

            with col_q:
                st.image(up, caption="Client image", use_container_width=True)

            with st.spinner("Removing background & searching..."):
                try:
                    emb     = get_embedding(up.read(), model, device)
                    results = search(emb, index)
                except Exception as e:
                    st.error(f"Search failed: {e}")
                    results = []

            with col_r:
                cols = st.columns(5)
                for i, (path, score) in enumerate(results[:5]):
                    with cols[i]:
                        try:
                            st.image(path, caption=f"{score:.1%} match", use_container_width=True)
                        except Exception:
                            st.caption("Image file not found")

            st.divider()

# ── Sidebar ───────────────────────────────────────────────
st.sidebar.title("Inventory Management")

if st.sidebar.button("🔄 Sync Inventory"):
    with st.spinner("Scanning new images... this may take several minutes."):
        exit_code = os.system("python build_index.py")
    if exit_code == 0:
        st.cache_data.clear()
        st.sidebar.success("Sync complete! Reload the page.")
    else:
        st.sidebar.error("Sync encountered errors. Check errors.txt.")

if st.sidebar.button("📋 View Error Log"):
    if os.path.exists(ERROR_LOG):
        with open(ERROR_LOG) as f:
            content = f.read().strip()
        if content:
            st.sidebar.text_area("errors.txt", content, height=250)
        else:
            st.sidebar.success("No errors logged.")
    else:
        st.sidebar.success("No error log found.")