# import os
# import json
# from tqdm import tqdm
# import torch
# from PIL import Image
# from transformers import CLIPProcessor, CLIPModel

# # 加载 CLIP 模型
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# # 配置参数
# INPUT_JSON = r"D:\paper\FCDiffusion_code-main\datasets\training_data.json"  # 输入的 JSON 文件路径
# OUTPUT_JSON = "filtered_data.json"  # 过滤后的 JSON 文件路径
# THRESHOLD = 0.4  # 余弦相似度的过滤阈值

# def calculate_clip_similarity(image_path, text):
#     """
#     计算单个图文对的 CLIP 余弦相似度。
#     """
#     try:
#         # 加载图像和文本
#         image = Image.open(image_path).convert("RGB")
#         inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True).to(device)
        
#         # 获取图像和文本的嵌入
#         outputs = model(**inputs)
#         image_embeds = outputs.image_embeds
#         text_embeds = outputs.text_embeds

#         # 计算余弦相似度
#         similarity = torch.cosine_similarity(image_embeds, text_embeds).item()
#         return similarity
#     except Exception as e:
#         print(f"Error processing {image_path}: {e}")
#         return None

# def filter_dataset(input_json, output_json, threshold):
#     """
#     过滤数据集，移除余弦相似度低于指定阈值的图文对。
#     """
#     # 读取数据
#     with open(input_json, "r", encoding="utf-8") as f:
#         data = json.load(f)
    
#     filtered_data = []

#     for item in tqdm(data, desc="Processing pairs"):
#         image_path = item.get("image_path")
#         text = item.get("text")
        
#         # 跳过无效的图像路径或文本
#         if not image_path or not text:
#             continue
        
#         # 计算余弦相似度
#         similarity = calculate_clip_similarity(image_path, text)
#         if similarity is not None and similarity >= threshold:
#             # 将相似度附加到数据中
#             item["similarity"] = similarity
#             filtered_data.append(item)
    
#     # 保存过滤后的数据
#     with open(output_json, "w", encoding="utf-8") as f:
#         json.dump(filtered_data, f, indent=4, ensure_ascii=False)
    
#     print(f"Filtered dataset saved to {output_json}")

# # 执行过滤
# if __name__ == "__main__":
#     filter_dataset(INPUT_JSON, OUTPUT_JSON, THRESHOLD)


import os
import torch
import lpips
from torchvision import transforms
from PIL import Image

# Initialize LPIPS model (using AlexNet)
loss_fn = lpips.LPIPS(net='alex')

# Path to the folders
folder1 = r'D:\paper\results\file'  # Ground truth folder
folder2 = 'path_to_folder2'  # Generated images folder

def calculate_lpips(image1, image2):
    image1 = transforms.ToTensor()(image1).unsqueeze(0).cuda()  # Convert to tensor and move to GPU
    image2 = transforms.ToTensor()(image2).unsqueeze(0).cuda()
    return loss_fn(image1, image2).item()

# Loop through images in folder1 and folder2
for img_name in os.listdir(folder1):
    if img_name in os.listdir(folder2):  # Ensure the image exists in both folders
        image1 = Image.open(os.path.join(folder1, img_name))
        image2 = Image.open(os.path.join(folder2, img_name))

        # Calculate LPIPS score
        lpips_score = calculate_lpips(image1, image2)
        print(f"LPIPS score for {img_name}: {lpips_score}")
