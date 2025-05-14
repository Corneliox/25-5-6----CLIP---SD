import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# Initialize model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Settings
positive_prompt = "young, friendly, and energetic south east asian college girl"
negative_prompt = "bad anatomy, bad hands, three hands, three legs, bad arms, missing legs, missing arms, poorly drawn face, poorly rendered hands, bad face, fused face, cloned face, worst face, three crus, extra crus, fused crus, worst feet, three feet, fused feet, fused thigh, three thigh, extra thigh, worst thigh, missing fingers, extra fingers, ugly fingers, long fingers, bad composition, horn, extra eyes, huge eyes, 2girl, amputation, disconnected limbs, cartoon, cg, 3d, unreal, animate, cgi, render, artwork, illustration, 3d render, cinema 4d, artstation, octane render, mutated body parts, painting, oil painting, 2d, sketch, bad photography, bad photo, deviant art, aberrations, abstract, anime, black and white, collapsed, conjoined, creative, drawing, extra windows, harsh lighting, jpeg artifacts, low saturation, monochrome, multiple levels, overexposed, oversaturated, photoshop, rotten, surreal, twisted, UI, underexposed, unnatural, unreal engine, unrealistic, video game, deformed body features"
image_folders = {
    "Enhance": "./Dataset/Train/2_Enhance",
    "Dzine": "./Dataset/Dzine",
    "Sketch": "./Dataset/Sketch",
}
keywords = ["friendly", "young", "energetic"]  # Key positive prompt keywords

def calculate_clip_score(image_path, positive_prompt, negative_prompt, processor, model, device):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=[positive_prompt, negative_prompt], images=image, return_tensors="pt", padding=True).to(device)
    outputs = model(**inputs)
    # The higher score is better for positive prompt, the lower is better for negative prompt.
    positive_score = outputs.logits_per_image[0][0].item()
    negative_score = outputs.logits_per_image[0][1].item()
    return positive_score - negative_score # A simple way to combine.  You might want to experiment with different combinations.

# Process images from all specified folders
all_results = {}
for folder_name, image_folder in image_folders.items():
    results = []
    for filename in os.listdir(image_folder):
        if not (filename.endswith(".jpg") or filename.endswith(".png")):  # Adjust as needed
            continue
        image_path = os.path.join(image_folder, filename)
        score = calculate_clip_score(image_path, positive_prompt, negative_prompt, processor, model, device)
        results.append((filename, score))
        print(f"Folder: {folder_name}, Image: {filename}, Combined CLIPScore = {score:.4f}")

    results.sort(key=lambda x: x[1], reverse=True)  # Sort by combined score
    best = results[0] if results else ("N/A", "N/A")
    print(f"\n✅ Best Image in Folder {folder_name}: {best[0]} with Combined CLIPScore = {best[1]:.4f}")
    all_results[folder_name] = {"best_image": best[0], "best_score": best[1], "all_scores": results}

print("\n--- Keyword Analysis (Example - best image from Enhance folder) ---")
if "Enhance" in all_results:
    best_enhance_image = all_results["Enhance"]["best_image"]
    image_path = os.path.join(image_folders["Enhance"], best_enhance_image)
    for keyword in keywords:
        keyword_score = calculate_clip_score(image_path, keyword, negative_prompt, processor, model, device)
        print(f"  Keyword '{keyword}': Score = {keyword_score:.4f}")

# You can adapt the keyword analysis for other folders as needed.
print("\nOverall Results:")
for folder, data in all_results.items():
    print(f"  Folder: {folder}, Best Image: {data['best_image']}, Best Score: {data['best_score']}")