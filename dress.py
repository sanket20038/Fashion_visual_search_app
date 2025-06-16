import os
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
import torch
import numpy as np
from torchvision import models, transforms
from tqdm import tqdm

# Config
CSV_PATH = "data/dresses_bd_processed_data.csv"
IMAGE_DIR = "dresses_images"
EMBEDDING_FILE = "dresses_image_embeddings.npy"
METADATA_FILE = "dresses_metadata.csv"

# Create image directory
os.makedirs(IMAGE_DIR, exist_ok=True)

# Load CSV
df = pd.read_csv(CSV_PATH)

# Download images function
def download_image(url, pid):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content)).convert("RGB")
            path = os.path.join(IMAGE_DIR, f"{pid}.jpg")
            img.save(path)
            return path
    except Exception as e:
        print(f"Failed to download {url}: {e}")
    return None

print("Downloading images...")
df['image_path'] = df.apply(lambda row: download_image(row['feature_image_s3'], row['product_id']), axis=1)

# Filter rows with valid images
valid_df = df[df['image_path'].notnull()].reset_index(drop=True)

# Load ResNet50 model
print("Loading ResNet50 model...")
model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])  # remove final fc
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Extract embeddings
embeddings = []
print("Extracting embeddings...")
for img_path in tqdm(valid_df['image_path']):
    try:
        img = Image.open(img_path).convert("RGB")
        img_t = transform(img).unsqueeze(0)
        with torch.no_grad():
            emb = model(img_t).squeeze().numpy()
        emb /= np.linalg.norm(emb)
        embeddings.append(emb)
    except Exception as e:
        print(f"Failed to process {img_path}: {e}")
        embeddings.append(np.zeros(2048))

# Save embeddings and metadata
print(f"Saving embeddings to {EMBEDDING_FILE} and metadata to {METADATA_FILE}...")
np.save(EMBEDDING_FILE, np.array(embeddings))
valid_df.to_csv(METADATA_FILE, index=False)

print("Done!")

