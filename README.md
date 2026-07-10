# Jewelry Visual Search Engine

A local reverse image search engine for a diamond jewelry showroom. Upload or capture a reference jewelry image and find visually similar items from the inventory catalog.

## Features
- Visual similarity search using Meta DINOv2 (ViT-L/14)
- Background removal using rembg for cleaner matching
- Cosine similarity search across indexed jewelry catalog
- Streamlit web UI for easy image upload and search
- Fully local — no cloud, no data leaves the system
- Docker support for easy deployment

## Tech Stack
- Model: Meta DINOv2 ViT-L/14 (vision transformer)
- Backend: Python, Flask
- UI: Streamlit
- Vector Storage: pickle-based NumPy vector store
- Similarity: Cosine similarity search
- Containerization: Docker

## Setup

### Without Docker:
pip install -r requirements.txt
python build_index.py
python app.py

### With Docker:
docker-compose up --build

## How it Works
1. Build index — DINOv2 extracts 1024-dim embeddings from all inventory images
2. User uploads query image
3. System extracts embedding from query image
4. Cosine similarity computed against all indexed embeddings
5. Top matching jewelry items returned!

## Dataset
~486 jewelry images indexed from showroom inventory.
