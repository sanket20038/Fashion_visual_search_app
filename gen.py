import os
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
import torch
import numpy as np
from torchvision import models, transforms
from tqdm import tqdm

# ========== CONFIG ==========
CSV_PATH = "data/jeans_bd_processed_data.csv"  # or replace with dresses file
IMAGE_DIR = "images"
EMBEDDING_FILE = "image_embeddings.npy"
FILTERED_CSV = "products_metadata.csv"
# ============================

# Step 1: Load CSV
df = pd.read_csv(CSV_PATH)
os.makedirs(IMAGE_DIR, exist_ok=True)

# Step 2: Download Images
def download_image(url, pid):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content)).convert("RGB")
            path = os.path.join(IMAGE_DIR, f"{pid}.jpg")
            img.save(path)
            return path
    except:
        pass
    return None

print("📥 Downloading images...")
df['image_path'] = df.apply(lambda row: download_image(row['feature_image_s3'], row['product_id']), axis=1)
df = df[df['image_path'].notnull()].reset_index(drop=True)

# Step 3: Load ResNet50 model
print("🧠 Loading ResNet50...")
model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])  # remove final FC
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Step 4: Extract Embeddings
print("🔍 Extracting image embeddings...")
embeddings = []

for path in tqdm(df['image_path']):
    try:
        img = Image.open(path).convert("RGB")
        img_t = transform(img).unsqueeze(0)
        with torch.no_grad():
            vec = model(img_t).squeeze().numpy()
            vec = vec / np.linalg.norm(vec)  # normalize
            embeddings.append(vec)
    except:
        embeddings.append(np.zeros(2048))  # fallback if image fails

# Step 5: Save
np.save(EMBEDDING_FILE, np.array(embeddings))
df.to_csv(FILTERED_CSV, index=False)
print("✅ Embeddings saved to:", EMBEDDING_FILE)
print("✅ Metadata saved to:", FILTERED_CSV)
