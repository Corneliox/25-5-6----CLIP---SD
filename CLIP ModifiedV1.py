import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# Initialize model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Settings
prompt = "young, friendly, and energetic south east asian college girl"
image_folder = "/content/drive/MyDrive/MindyClip/Image/Enhanced"
avoid_keywords = ["Dzine", "Ilustration"]

# Iterate over image versions (assuming images named 5.jpg ... 15.jpg or 5.png, etc.)
results = []
for i in range(5, 16):
    image_path = os.path.join(image_folder, f"{i}.jpg")  # modify if .png or else
    if not os.path.exists(image_path):
        continue
    if any(skip in image_path for skip in avoid_keywords):
        continue

    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=prompt, images=image, return_tensors="pt", padding=True).to(device)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    score = logits_per_image.softmax(dim=1).item()

    results.append((i, score))
    print(f"Iteration {i}: CLIPScore = {score:.4f}")

# Sort and find best
results.sort(key=lambda x: x[1], reverse=True)
best = results[0]
print(f"\n✅ Best Iteration: {best[0]} with CLIPScore = {best[1]:.4f}")
