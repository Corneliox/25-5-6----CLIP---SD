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
image_folder = "./Dataset/Train\2_Enhance"
avoid_keywords = ["bad_anatomy,bad_hands,three_hands,three_legs,bad_arms,missing_legs,missing_arms,poorly_drawn_face,poorly_rendered_hands,bad_face,fused_face,cloned_face,worst_face,three_crus,extra_crus,fused_crus,worst_feet,three_feet,fused_feet,fused_thigh,three_thigh,extra_thigh,worst_thigh,missing_fingers,extra_fingers,ugly_fingers,long_fingers,bad_composition,horn,extra_eyes,huge_eyes,2girl,amputation,disconnected_limbs,cartoon,cg,3d,unreal,animate,cgi,render,artwork,illustration,3d_render,cinema_4d,artstation,octane_render,mutated_body_parts,painting,oil_painting,2d,sketch,bad_photography,bad_photo,deviant_art,aberrations,abstract,anime,black_and_white,collapsed,conjoined,creative,drawing,extra_windows,harsh_lighting,jpeg_artifacts,low_saturation,monochrome,multiple_levels,overexposed,oversaturated,photoshop,rotten,surreal,twisted,UI,underexposed,unnatural,unreal_engine,unrealistic,video_game,deformed_body_features"]

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
