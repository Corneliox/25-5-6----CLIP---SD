# CLIPing on VID Unika, for image generating prompting
Taraaaa


# 🎯 Goal Recap: Kohya Training for CLIP Evaluation

This README describes how to fine-tune Stable Diffusion 1.5 using Kohya on a low-VRAM GPU (RTX 2050 4GB) and evaluate the quality of image outputs using CLIPScore. The goal is to determine the best iteration that produces images aligned with the prompt:

> “**young, friendly, and energetic south east asian college girl**”

---

## ✅ Objective

Train on images in `/Enhance` while **ignoring** `/Dzine` and `/Sketch`, then evaluate outputs using CLIPScore to find which **iteration (5–15)** produces the best result.

---

## 🗂 Dataset Structure

```
/Dataset/
    /Enhance/       <- All images tagged using MD14
    /Dzine/
    /Sketch/
```

Only `/Enhance` is used for training.

---

## 🧠 Kohya Setup

### 🔹 Pretrained Model

Use either of:
- `xzl.safetensors` (38 MB)
- `2023autumn.safetensors` (147 MB)

### 🔹 Basic Configuration

| Parameter             | Value                        |
|----------------------|------------------------------|
| Model                 | SD 1.5 base (use above)      |
| Dataset Directory     | `/training_dataset/Enhance`  |
| Output Directory      | `/output_lora`               |
| Caption Extension     | `.txt`                       |
| Use 8-bit Adam        | ✅                           |
| Gradient Accumulation | `2` or `4`                   |
| Precision             | `fp16`                       |

---

### 🔹 LoRA Settings (Recommended for 4GB GPU)

| Parameter     | Value     |
|---------------|-----------|
| Rank          | `4` or `8`|
| Alpha         | Same as Rank |
| Resolution    | `512x512` |
| Shuffle Captions | ✅     |
| Save Frequency | Every 1 Epoch |
| Learning Rate | `1e-4` to `2e-4` |
| Epochs        | `5–15`    |
| Batch Size    | `1`       |

---

## 🧪 Prompt Testing After Training

Prompt:
```
"young, friendly, and energetic south east asian college girl"
```

Sampling settings:
- CFG Scale: 7–8
- Steps: 25
- Sampler: DPM++ 2M or Euler a
- Fixed Seed for consistent comparison

---

## 📊 Post-Training Evaluation (CLIPScore)

1. Load each saved model (e.g. `epoch-5.safetensors`, `epoch-6.safetensors`, etc.)
2. Generate image from prompt
3. Evaluate output using CLIPScore
4. Determine best iteration (highest alignment score)

---

## 💡 Tips for Low VRAM (RTX 2050)

- Enable `--xformers` if supported
- Use **LoRA**, not full DreamBooth
- Keep batch size at 1
- Use gradient accumulation = 4

---

## 📁 Output

You should now have:
- Trained LoRA models
- Images generated per iteration
- A CLIPScore report showing which iteration matches the prompt best

---
