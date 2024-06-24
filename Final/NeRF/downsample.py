import os
from PIL import Image
 
# 输入文件夹路径和输出文件夹路径
input_folder = "/root/autodl-tmp/nerf-pytorch/data/nerf_llff_data/kipling/images"
output_folder = "/root/autodl-tmp/nerf-pytorch/data/nerf_llff_data/kipling/images_8"
 
# 创建输出文件夹
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
 
# 获取输入文件夹中所有图片文件的列表
image_files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
 
# 循环处理每个图片文件
for image_file in image_files:
    # 打开图像文件
    image_path = os.path.join(input_folder, image_file)
    image = Image.open(image_path)
 
    # 获取原始图像的宽度和高度
    width, height = image.size
 
    # 计算新图像的宽度和高度（原始图像的1/8）
    new_width = width // 8
    new_height = height // 8
 
    # 使用resize()函数对图像进行下采样
    downscaled_image = image.resize((new_width, new_height), Image.ANTIALIAS)
 
    # 构造输出文件路径
    output_path = os.path.join(output_folder, image_file)
 
    # 保存下采样后的图像
    downscaled_image.save(output_path)
 
    print(f"Downsampling complete: {image_file}")
 
print("All images downscaled successfully.")