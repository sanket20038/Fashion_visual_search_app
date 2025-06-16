import faiss
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import models, transforms
import torch
import os

# ========== CONFIG ==========
EMBEDDING_FILE = "image_embeddings.npy"
METADATA_CSV = "products_metadata.csv"
TOP_K = 5
# ============================

# Load embeddings and metadata
embeddings = np.load(EMBEDDING_FILE)
df = pd.read_csv(METADATA_CSV)

# Build FAISS index
print("🔍 Building FAISS index...")
d = embeddings.shape[1]  # embedding dimension
index = faiss.IndexFlatL2(d)
index.add(embeddings)

# Load ResNet50 model for query image
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

# Function to extract embedding from image
def extract_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    img_t = transform(image).unsqueeze(0)
    with torch.no_grad():
        vec = model(img_t).squeeze().numpy()
        vec = vec / np.linalg.norm(vec)
    return vec

# Query image path
query_path = input("Enter path to query image: ").strip()

# Compute embedding and search
print("🔎 Searching for similar items...")
query_vec = extract_embedding(query_path).astype("float32").reshape(1, -1)
distances, indices = index.search(query_vec, TOP_K)

# Show results
print(f"\n🎯 Top {TOP_K} similar products:\n")
for idx in indices[0]:
    print(f"- {df.iloc[idx]['product_name']} ({df.iloc[idx]['brand']})")
    print(f"  URL: {df.iloc[idx]['pdp_url']}")
    print(f"  Image: {df.iloc[idx]['feature_image_s3']}\n")
