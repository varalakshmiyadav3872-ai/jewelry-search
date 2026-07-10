import io
import os
import json
import pickle
import hashlib
import shutil
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from PIL import Image
from rembg import remove
from torchvision import transforms

# ── Config ──────────────────────────────────────────────
INVENTORY_DIR = "/app/inventory_images"
INDEX_DIR     = "/app/index"
BACKUP_DIR    = "/app/backups"
INDEX_FILE    = f"{INDEX_DIR}/jewelry_index.pkl"
PROGRESS_FILE = f"{INDEX_DIR}/progress.json"
ERROR_LOG     = f"{INDEX_DIR}/errors.txt"
BATCH_SIZE    = 50
MAX_BACKUPS   = 5

os.makedirs(INDEX_DIR, exist_ok=True)
os.makedirs(BACKUP_DIR, exist_ok=True)

# ── Model ────────────────────────────────────────────────
print("Loading DINOv2 model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
model.eval().to(device)
print(f"Running on: {device}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ── Helpers ──────────────────────────────────────────────
def get_sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def get_embedding(image_path):
    with open(image_path, "rb") as f:
        raw = f.read()
    cleaned = remove(raw)                                   # strip background
    img     = Image.open(io.BytesIO(cleaned)).convert("RGB")
    tensor  = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(tensor).squeeze().cpu().numpy()
    return emb

def log_error(msg):
    with open(ERROR_LOG, "a") as f:
        f.write(f"{datetime.now()} | {msg}\n")

# ── Index I/O ────────────────────────────────────────────
def backup_index():
    if os.path.exists(INDEX_FILE):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        shutil.copy2(INDEX_FILE, f"{BACKUP_DIR}/jewelry_index_{ts}.pkl")
        backups = sorted(Path(BACKUP_DIR).glob("*.pkl"))
        for old in backups[:-MAX_BACKUPS]:
            old.unlink()

def save_index(index):
    backup_index()
    tmp = INDEX_FILE + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(index, f)
    os.replace(tmp, INDEX_FILE)           # atomic swap
    print(f"  Saved {len(index)} items to index")

def load_index():
    if os.path.exists(INDEX_FILE):
        with open(INDEX_FILE, "rb") as f:
            return pickle.load(f)
    return {}

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return set(json.load(f)["done"])
    return set()

def save_progress(done):
    with open(PROGRESS_FILE, "w") as f:
        json.dump({"done": list(done)}, f)

# ── Main ─────────────────────────────────────────────────
def build_index():
    index = load_index()
    done  = load_progress()
    existing = set(index.keys())

    # collect all images
    images = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
        images.extend(Path(INVENTORY_DIR).rglob(ext))
    print(f"Found {len(images)} images in inventory")

    # filter to only new ones
    new = []
    for p in images:
        try:
            sha = get_sha256(str(p))
            if sha not in existing and sha not in done:
                new.append((p, sha))
        except Exception as e:
            log_error(f"Hash error {p}: {e}")

    print(f"{len(new)} new images to process")
    if not new:
        print("Nothing to do.")
        return

    total_batches = (len(new) - 1) // BATCH_SIZE + 1
    for i in range(0, len(new), BATCH_SIZE):
        batch = new[i:i + BATCH_SIZE]
        print(f"Batch {i//BATCH_SIZE + 1}/{total_batches} ...")
        for p, sha in batch:
            try:
                emb = get_embedding(str(p))
                index[sha] = {"path": str(p), "embedding": emb}
                done.add(sha)
            except Exception as e:
                log_error(f"Embed error {p}: {e}")

        save_index(index)       # save after every batch
        save_progress(done)

    # clean up progress file when fully done
    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)

    print("\nDone! Check errors.txt for any skipped images.")

if __name__ == "__main__":
    build_index()