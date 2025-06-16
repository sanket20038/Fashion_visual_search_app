from flask import Flask, render_template, request
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import faiss
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import models, transforms

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load jeans
jeans_metadata = pd.read_csv("products_metadata.csv")
jeans_embeddings = np.load("image_embeddings.npy").astype("float32")

# Load dresses
dresses_metadata = pd.read_csv("dresses_metadata.csv")
dresses_embeddings = np.load("dresses_image_embeddings.npy").astype("float32")

dimension = jeans_embeddings.shape[1]

# Build FAISS indexes
jeans_index = faiss.IndexFlatL2(dimension)
jeans_index.add(jeans_embeddings)

dresses_index = faiss.IndexFlatL2(dimension)
dresses_index.add(dresses_embeddings)

# Load model
model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def extract_embedding(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        img_t = transform(image).unsqueeze(0)
        with torch.no_grad():
            vec = model(img_t).squeeze().numpy()
        return vec / np.linalg.norm(vec)
    except Exception as e:
        print(f"[❌ ERROR] Failed to process image: {e}")
        return None

def detect_category(filename):
    fname = filename.lower()
    if any(x in fname for x in ['dress', 'top']):
        return "dress"
    return "jeans"

def recommend_complementary(item, item_category):
    if item_category == "jeans":
        # Recommend top for jeans
        brand = item['brand']
        filtered = dresses_metadata[dresses_metadata['brand'] == brand]
        if filtered.empty:
            filtered = dresses_metadata

        index = faiss.IndexFlatL2(dresses_embeddings.shape[1])
        embeddings = dresses_embeddings[filtered.index]
        index.add(embeddings)

        item_idx = jeans_metadata.index[jeans_metadata['product_id'] == item['product_id']].tolist()[0]
        query_vec = jeans_embeddings[item_idx].reshape(1, -1)
        _, idxs = index.search(query_vec, 1)
        rec = filtered.iloc[idxs[0][0]]
    else:
        # Recommend jeans for dress
        brand = item['brand']
        filtered = jeans_metadata[jeans_metadata['brand'] == brand]
        if filtered.empty:
            filtered = jeans_metadata

        index = faiss.IndexFlatL2(jeans_embeddings.shape[1])
        embeddings = jeans_embeddings[filtered.index]
        index.add(embeddings)

        item_idx = dresses_metadata.index[dresses_metadata['product_id'] == item['product_id']].tolist()[0]
        query_vec = dresses_embeddings[item_idx].reshape(1, -1)
        _, idxs = index.search(query_vec, 1)
        rec = filtered.iloc[idxs[0][0]]

    return {
        "name": rec['product_name'],
        "brand": rec['brand'],
        "url": rec['pdp_url'],
        "image": rec['feature_image_s3']
    }

@app.route("/", methods=["GET", "POST"])
def home():
    results = []
    query_img_path = None
    category = None

    if request.method == "POST":
        file = request.files["image"]
        if file:
            query_img_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(query_img_path)

            query_vec = extract_embedding(query_img_path)
            if query_vec is None:
                return "Failed to process uploaded image. Please try another.", 400

            category = detect_category(file.filename)
            query_vec = query_vec.astype("float32").reshape(1, -1)

            if category == "jeans":
                _, indices = jeans_index.search(query_vec, 5)
                for idx in indices[0]:
                    item = jeans_metadata.iloc[idx]
                    recommendation = recommend_complementary(item, "jeans")
                    results.append({
                        "name": item['product_name'],
                        "brand": item['brand'],
                        "url": item['pdp_url'],
                        "image": item['feature_image_s3'],
                        "recommendation": recommendation
                    })
            else:
                _, indices = dresses_index.search(query_vec, 5)
                for idx in indices[0]:
                    item = dresses_metadata.iloc[idx]
                    recommendation = recommend_complementary(item, "dress")
                    results.append({
                        "name": item['product_name'],
                        "brand": item['brand'],
                        "url": item['pdp_url'],
                        "image": item['feature_image_s3'],
                        "recommendation": recommendation
                    })

    return render_template("index.html", results=results, query=query_img_path, category=category)

if __name__ == "__main__":
    app.run(debug=True)
