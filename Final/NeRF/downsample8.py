import os
from PIL import Image


input_folder = "/root/autodl-tmp/nerf-pytorch/data/nerf_llff_data/kipling/images"
output_folder = "/root/autodl-tmp/nerf-pytorch/data/nerf_llff_data/kipling/images_8"
 
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
 
image_files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
 
for image_file in image_files:
    image_path = os.path.join(input_folder, image_file)
    image = Image.open(image_path)
 
    width, height = image.size
 
    new_width = width // 8
    new_height = height // 8
 
    downscaled_image = image.resize((new_width, new_height), Image.ANTIALIAS)
    
    output_path = os.path.join(output_folder, image_file)
 
    downscaled_image.save(output_path)
 
    print(f"Downsampling complete: {image_file}")
 
print("All images downscaled successfully.")