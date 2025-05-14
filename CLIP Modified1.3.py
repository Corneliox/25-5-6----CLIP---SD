import os
import torch
from PIL import Image
from torchvision import transforms
import open_clip

# Load CLIP model
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
tokenizer = open_clip.get_tokenizer('ViT-B-32')
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device).eval()

# Prompts
positive_prompt = "young, friendly, and energetic south east asian college girl"
negative_prompt = (
    "bad_anatomy,bad_hands,three_hands,three_legs,bad_arms,missing_legs,missing_arms,"
    "poorly_drawn_face,poorly_rendered_hands,bad_face,fused_face,cloned_face,worst_face,"
    "three_crus,extra_crus,fused_crus,worst_feet,three_feet,fused_feet,fused_thigh,three_thigh,"
    "extra_thigh,worst_thigh,missing_fingers,extra_fingers,ugly_fingers,long_fingers,bad_composition,"
    "horn,extra_eyes,huge_eyes,2girl,amputation,disconnected_limbs,cartoon,cg,3d,unreal,animate,cgi,"
    "render,artwork,illustration,3d_render,cinema_4d,artstation,octane_render,mutated_body_parts,painting,"
    "oil_painting,2d,sketch,bad_photography,bad_photo,deviant_art,aberrations,abstract,anime,"
    "black_and_white,collapsed,conjoined,creative,drawing,extra_windows,harsh_lighting,jpeg_artifacts,"
    "low_saturation,monochrome,multiple_levels,overexposed,oversaturated,photoshop,rotten,surreal,"
    "twisted,UI,underexposed,unnatural,unreal_engine,unrealistic,video_game,deformed_body_features"
)

# Tokenize prompts
positive_tokens = tokenizer([positive_prompt]).to(device)
negative_tokens = tokenizer([negative_prompt]).to(device)

# Encode prompts
with torch.no_grad():
    positive_embed = model.encode_text(positive_tokens).float()
    negative_embed = model.encode_text(negative_tokens).float()

# Normalize embeddings
positive_embed /= positive_embed.norm(dim=-1, keepdim=True)
negative_embed /= negative_embed.norm(dim=-1, keepdim=True)

# Folder path
image_folder = "./Dataset/Train/2_Enhance"

# Evaluate images
results = []
for filename in os.listdir(image_folder):
    if filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
        image_path = os.path.join(image_folder, filename)
        image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)

        with torch.no_grad():
            image_embed = model.encode_image(image).float()
            image_embed /= image_embed.norm(dim=-1, keepdim=True)

            # Compute similarities
            positive_score = (image_embed @ positive_embed.T).item()
            negative_score = (image_embed @ negative_embed.T).item()

            total_score = positive_score - negative_score
            results.append((filename, total_score, positive_score, negative_score))

# Sort results by total_score
results.sort(key=lambda x: x[1], reverse=True)

# Print top images
print(f"{'Image':<40} {'Score':>8} {'Pos':>8} {'Neg':>8}")
for name, score, pos, neg in results:
    print(f"{name:<40} {score:8.3f} {pos:8.3f} {neg:8.3f}")


GOOD_THRESHOLD = 0.2
output_dir = "./Dataset/Train/3_Good"
os.makedirs(output_dir, exist_ok=True)

for name, score, _, _ in results:
    if score >= GOOD_THRESHOLD:
        src = os.path.join(image_folder, name)
        dst = os.path.join(output_dir, name)
        Image.open(src).save(dst)
print(f"Images with score >= {GOOD_THRESHOLD} saved to {output_dir}")