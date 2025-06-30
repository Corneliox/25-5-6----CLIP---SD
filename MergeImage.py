from PIL import Image, ImageDraw, ImageFont
import os

def create_combined_image(image_folder, output_image_name,  font_size=12, text_padding=10, max_width=600, max_height=900):
    """
    Combines multiple images from a folder into a single image with filenames as labels,
    without resizing the input images, and creates a final image with dimensions close to
    the specified max_width and max_height while maintaining a 2:3 aspect ratio.

    Args:
        image_folder (str): Path to the folder containing the images.
        output_image_name (str): Name of the output image file.
        font_size (int, optional): Font size for the filename labels. Defaults to 12.
        text_padding (int, optional): Padding between the image and the text. Defaults to 10.
        max_width (int, optional): Maximum width of the final combined image. Defaults to 600.
        max_height (int, optional): Maximum height of the final combined image. Defaults to 900.

    Returns:
        None: Saves the combined image to the specified output file.  Prints success or error message.
    """
    # Cari semua file gambar di folder
    try:
        image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    except FileNotFoundError:
        print(f"Error: Folder not found at {image_folder}")
        return
    image_files.sort()  # Urutkan berdasarkan nama file

    # Pastikan ada gambar
    if not image_files:
        print(f"Error: No images found in {image_folder}")
        return

    # Determine maximum individual image width and height
    max_image_width = 0
    max_image_height = 0
    for img_file in image_files:
        img_path = os.path.join(image_folder, img_file)
        try:
            with Image.open(img_path) as img:
                max_image_width = max(max_image_width, img.width)
                max_image_height = max(max_image_height, img.height)
        except Exception as e:
            print(f"Warning: Failed to open {img_file} to get size. Skipping: {e}")
            return

    # Calculate aspect ratio and determine grid dimensions
    aspect_ratio = max_width / max_height  # Target aspect ratio (2:3)
    image_aspect = max_image_width / max_image_height

    if image_aspect > aspect_ratio:
        grid_width = int(aspect_ratio * len(image_files)**0.5)
        grid_height = int(len(image_files) / grid_width)
    else:
        grid_height = int(len(image_files)**0.5 / aspect_ratio)
        grid_width = int(len(image_files) / grid_height)
    grid_width = max(1, grid_width)
    grid_height = max(1, grid_height)
    # Adjust grid dimensions to fit all images
    while grid_width * grid_height < len(image_files):
        if grid_width < grid_height:
            grid_width += 1
        else:
            grid_height += 1
    # Use max_width and max_height to determine image_size, ensuring 2:3 ratio
    if max_width / max_height > aspect_ratio:
        image_size = max_height / grid_height
    else:
        image_size = max_width / grid_width
    image_size = int(image_size)
    # Calculate the final combined image dimensions based on the grid and image size
    total_width = int(grid_width * image_size)
    total_height = int(grid_height * (image_size + text_padding + font_size + 5))

    # Create the combined image
    combined_image = Image.new("RGB", (total_width, total_height), (255, 255, 255))

    # Load font
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except OSError:
        print("Warning: arial.ttf not found. Using default font.")
        font = ImageFont.load_default()

    # Add images to the combined image
    for idx, img_file in enumerate(image_files):
        if idx >= grid_width * grid_height:
            break
        row = idx // grid_width
        col = idx % grid_width

        img_path = os.path.join(image_folder, img_file)
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Warning: Failed to open {img_file}. Skipping: {e}")
            continue
        # Calculate position to center the image
        x = int(col * image_size + (image_size - img.width) / 2)
        y = int(row * (image_size + text_padding + font_size + 5) + (image_size - img.height) / 2)
        combined_image.paste(img, (x, y))

        # Add image name below
        text = os.path.splitext(img_file)[0]
        draw = ImageDraw.Draw(combined_image)
        try:
            text_width, text_height = draw.textsize(text, font=font)
        except AttributeError: # Handle the AttributeError
            text_width, text_height = draw.textsize(text, font=font)
        text_x = int(col * image_size + (image_size - text_width) / 2)
        text_y = int(row * (image_size + text_padding + font_size + 5) + image_size + text_padding)
        draw.text((text_x, text_y), text, fill="black", font=font)

    # Resize the combined image to the target resolution (600x900 or closest 2:3)
    combined_image = combined_image.resize((max_width, max_height), Image.LANCZOS)
    # Save the combined image
    try:
        combined_image.save(output_image_name)
        print(f"Combined image successfully saved as: {output_image_name}")
    except Exception as e:
        print(f"Error: Failed to save the combined image: {e}")



if __name__ == "__main__":
    # Konfigurasi
    IMAGE_FOLDER = "./Dataset/Train/2_Enhance"  # Ganti dengan path folder gambar Anda
    OUTPUT_IMAGE_NAME = "combined_54_images.png"
    FONT_SIZE = 12
    TEXT_PADDING = 10
    MAX_WIDTH = 600
    MAX_HEIGHT = 900
    create_combined_image(IMAGE_FOLDER, OUTPUT_IMAGE_NAME, FONT_SIZE, TEXT_PADDING, MAX_WIDTH, MAX_HEIGHT)
