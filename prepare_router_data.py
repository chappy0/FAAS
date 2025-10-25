import cv2
import numpy as np
import os
import json
import argparse
from tqdm import tqdm

def get_image_text_pairs(directory):

    image_files = {}
    text_files = {}
    

    for root, _, files in os.walk(directory):
        for file in files:
            filename_no_ext = os.path.splitext(file)[0]
            file_path = os.path.join(root, file)
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files[filename_no_ext] = file_path
            elif file.lower().endswith('.txt'):
                text_files[filename_no_ext] = file_path
                

    paired_data = []
    for name, img_path in image_files.items():
        if name in text_files:
            txt_path = text_files[name]
            try:
                with open(txt_path, 'r', encoding='utf-8') as f:
                    prompt = f.read().strip()
                    if prompt: 
                        paired_data.append({"image_path": img_path, "prompt": prompt})
            except Exception as e:
                print(f"Warning: Could not read or process {txt_path}: {e}")
                
    return paired_data

def calculate_edge_density(image_path, canny_threshold1=100, canny_threshold2=200):

    try:
        image = cv2.imread(image_path)
        if image is None:
            return None
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        

        edges = cv2.Canny(gray_image, threshold1=canny_threshold1, threshold2=canny_threshold2)
        

        edge_pixels = np.count_nonzero(edges)
        total_pixels = edges.shape[0] * edges.shape[1]
        
        complexity_score = edge_pixels / total_pixels if total_pixels > 0 else 0
        return complexity_score
    except Exception as e:
        print(f"Warning: Could not calculate complexity for {image_path}: {e}")
        return None

def main(args):
    print(f"Step 1: Finding image-text pairs in '{args.data_dir}'...")
    paired_data = get_image_text_pairs(args.data_dir)
    print(f"Found {len(paired_data)} pairs.")
    
    if not paired_data:
        print("No data found. Exiting.")
        return

    router_training_data = []
    
    print("Step 2: Calculating visual complexity for each image...")

    for item in tqdm(paired_data, desc="Processing images"):
        complexity_score = calculate_edge_density(item["image_path"])
        
        if complexity_score is not None:
            router_training_data.append({
                "prompt": item["prompt"],
                "complexity_score": complexity_score
            })
            
    print(f"Successfully processed {len(router_training_data)} items.")
    
    print(f"Step 3: Saving preprocessed data to '{args.output_file}'...")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(router_training_data, f, indent=4)
        
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data for training the Prompt Complexity Router.")
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the directory containing image-text pairs (e.g., your LAION subset).')
    parser.add_argument('--output_file', type=str, default='router_training_data.json', help='Path to save the output JSON file.')
    
    args = parser.parse_args()
    main(args)
