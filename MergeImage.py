from PIL import Image, ImageDraw, ImageFont
import os

# Konfigurasi
IMAGE_FOLDER = "./Dataset/Train/2_Enhance"  # Ganti dengan path folder gambar Anda
OUTPUT_IMAGE_NAME = "combined_54_images.png"
IMAGE_SIZE = 200  # Ukuran persegi tiap gambar (misalnya 200x200)
GRID_ROWS = 6
GRID_COLS = 9
FONT_SIZE = 12
TEXT_PADDING = 10

# Cari semua file gambar di folder
image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
image_files.sort()  # Urutkan berdasarkan nama file

# Pastikan ada minimal 54 gambar
if len(image_files) < 54:
    print(f"Error: Hanya {len(image_files)} gambar ditemukan. Butuh 54 gambar.")
    exit()

# Ambil 54 gambar pertama
selected_images = image_files[:54]

# Hitung ukuran total gambar
total_width = GRID_COLS * IMAGE_SIZE
total_height = GRID_ROWS * (IMAGE_SIZE + TEXT_PADDING)

# Buat canvas kosong
combined_image = Image.new("RGB", (total_width, total_height), (255, 255, 255))

# Load font untuk teks
try:
    font = ImageFont.truetype("arial.ttf", FONT_SIZE)
except:
    font = ImageFont.load_default()

# Tambahkan gambar ke canvas
for idx, img_file in enumerate(selected_images):
    row = idx // GRID_COLS
    col = idx % GRID_COLS

    img_path = os.path.join(IMAGE_FOLDER, img_file)
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"Gagal membuka {img_file}: {e}")
        continue

    # Resize gambar ke ukuran persegi
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Image.Resampling.LANCZOS)

    # Tempatkan gambar ke canvas
    x = col * IMAGE_SIZE
    y = row * (IMAGE_SIZE + TEXT_PADDING)
    combined_image.paste(img, (x, y))

    # Tambahkan nama file di bawah gambar
    text = os.path.splitext(img_file)[0]  # Tanpa ekstensi
    draw = ImageDraw.Draw(combined_image)
    text_width, text_height = draw.textsize(text, font=font)
    text_x = x + (IMAGE_SIZE - text_width) // 2
    text_y = y + IMAGE_SIZE + 5
    draw.text((text_x, text_y), text, fill="black", font=font)

# Simpan hasil akhir
combined_image.save(OUTPUT_IMAGE_NAME)
print(f"Gambar gabungan telah disimpan sebagai: {OUTPUT_IMAGE_NAME}")