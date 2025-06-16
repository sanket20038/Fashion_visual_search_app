# Fashion_visual_search_app

👗 Fashion Visual Search & Intelligent Styling Assistant

---

🏢 Industry Context

The global fashion e-commerce market loses up to **65% of potential customers** due to the inability to find specific products. Traditional text-based search fails to describe **nuanced visual details** like color gradients, patterns, textures that drive fashion decisions. This system aims to bridge that gap using visual AI.

---

## 🎯 Problem Statement

Build an end-to-end **machine learning system** that enables users to:

- Upload any fashion image (from social media, camera, wardrobe, etc.)
- Receive:
  - ✅ Exact or visually similar items from inventory
  - ✅ Outfit recommendations that complement the image

---

## 🛠️ Solution Overview

This project uses:
- **ResNet50** for image feature extraction
- **FAISS** for real-time similarity search
- **Flask** for the user interface
- **CSV & image URLs** as structured product data

Users can search by image and choose between categories (e.g., jeans, dresses). Results include product name, brand, image, and link.

---

## 🔍 Key Challenges Addressed

| Challenge                    | Solution                                                                 |
|-----------------------------|--------------------------------------------------------------------------|
| Visual Similarity at Scale  | FAISS indexing + ResNet embeddings                                       |
| Multi-Modal Understanding   | Uses metadata (brand, category) along with visual features               |
| Style Compatibility         | Prototype includes category-based filtering and outfit extensions        |
| Trend Awareness             | Dynamic data loading enables fresh recommendations                       |
| User Experience             | Mobile-friendly UI, category dropdown, drag-and-drop support             |

---

## 📦 Dataset Format

Sample columns used:
- `product_name`, `brand`, `description`, `category_id`, `feature_image_s3`
- `image_embeddings.npy` — vectors extracted from ResNet50
- `products_metadata.csv` — combined metadata for display
- `feature_list`, `style_attributes` — can be used for future outfit scoring

----------------------------------------------------------------------------------------------------------------------------------------

## ▶️ How to Run Locally

1. Install Python packages:
```bash
pip install -r requirements.txt

2. Start the Flask server:

```bash
python app.py
----

it will start at  http://127.0.0.1:5000

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
app.py — Flask backend

data - raw data which was provided in the problem statement
images - folder containing images of products of jeans
dresses_images - folder containing images of products of dresses
templates/index.html — Upload UI
image_embeddings.npy — Vector database 
products_metadata.csv — Metadata display
requirements.txt — Python libraries

------------------------------------------------------------------------------------------------------------------------------------
🔁 System Workflow
1. Image Upload (Frontend)
User visits the web app (Flask UI).
![alt text](image.png)

Uploads a fashion image (e.g., dress, jeans, top).
use sample image from this file to test the system sample.png

2. Image Preprocessing
The image is resized to 224×224 pixels.

It's converted into a tensor and normalized using torchvision.transforms.

3. Feature Extraction (ResNet50)
The image is passed through a pretrained ResNet50 model.

We remove the last layer to get a 2048-dimensional vector (embedding).

This vector represents the image’s visual features: color, shape, texture, pattern.

4. Similarity Search (FAISS)
A FAISS index is already built using embeddings from the product dataset.

The system compares the uploaded image’s vector with all other vectors in the index.

It finds the Top 5 most visually similar products based on L2 distance.

5. Product Matching
Each match includes:

product_name

brand

image URL (for preview)

pdp_url (Product Detail Page)

6. Output on Web UI
The uploaded image is displayed back to the user.

The system shows Top 5 similar products with image, name, and link.

demo video 
<video controls src="demo.mp4" title="Demo"></video>