import torch
import numpy as np
import argparse
from collections import defaultdict
from omegaconf import OmegaConf
from tqdm import tqdm
import os
import json

from ldm.util import instantiate_from_config
from torch.utils.data import DataLoader
from fcdiffusion.dataset import TestDataset
from fcdiffusion.fcdiffusion import FCDiffusion, ConditionallySkipableBlock
from tools.dct_util import dct_2d

captured_activations = {}


def get_image_text_pairs(directory):
    image_files, text_files, paired_data = {}, {}, []
    print(f"Scanning directory '{directory}' for image-text pairs...")
    for root, _, files in os.walk(directory):
        for file in files:
            filename_no_ext = os.path.splitext(file)[0]
            file_path = os.path.join(root, file)
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files[filename_no_ext] = file_path
            elif file.lower().endswith('.txt'):
                text_files[filename_no_ext] = file_path
                
    for name, img_path in image_files.items():
        if name in text_files:
            try:
                with open(text_files[name], 'r', encoding='utf-8') as f:
                    prompt = f.read().strip()
                    if prompt: paired_data.append({"image_path": img_path, "prompt": prompt})
            except Exception as e:
                print(f"Warning: Could not read {text_files[name]}: {e}")
                
    print(f"Found {len(paired_data)} pairs.")
    return paired_data

def get_activation(name):
    def hook(model, input, output):
        final_output = output[0].detach() if isinstance(output, tuple) else output.detach()
        captured_activations[name] = final_output
    return hook


def make_frequency_weight(h, w, band, device):
    yy, xx = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
    center_y, center_x = h // 2, w // 2
    rr = torch.sqrt((yy - center_y)**2 + (xx - center_x)**2) / (0.5 * torch.sqrt(torch.tensor(h**2 + w**2, device=device)))
    
    if band == "mini_pass": mask = (rr < 0.15).float()
    elif band == "low_pass": mask = (rr < 0.35).float()
    elif band == "mid_pass": mask = ((rr >= 0.35) & (rr < 0.65)).float()
    elif band == "high_pass": mask = (rr >= 0.65).float()
    else: mask = torch.ones_like(rr)
    return mask


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = OmegaConf.load(args.config)
    config.model.params.control_mode = args.band
    model = instantiate_from_config(config.model)
    model.load_and_migrate_checkpoint(args.ckpt)
    model.to(device).eval()

    diffusion_model = model.model.diffusion_model

    hook_handles = []
    skippable_layer_names = []
    for i, block in enumerate(diffusion_model.input_blocks):
        if hasattr(block, '__iter__'):
            for j, sub_layer in enumerate(block):
                if isinstance(sub_layer, ConditionallySkipableBlock):
                    name = f"input_blocks.{i}.{j}"
                    handle = sub_layer.register_forward_hook(get_activation(name))
                    hook_handles.append(handle)
                    skippable_layer_names.append(name)
    
    for i, sub_layer in enumerate(diffusion_model.middle_block):
        if isinstance(sub_layer, ConditionallySkipableBlock):
            name = f"middle_block.{i}"
            handle = sub_layer.register_forward_hook(get_activation(name))
            hook_handles.append(handle)
            skippable_layer_names.append(name)

    image_text_pairs = get_image_text_pairs(args.data_dir)
    if not image_text_pairs: print("No image-text pairs found. Exiting."); return

    layer_stats = defaultdict(lambda: {"count": 0, "scores": []})

    print(f"Analyzing {args.num_samples} samples for diffusion_model...")
    with torch.no_grad():
        for i, item in enumerate(tqdm(image_text_pairs, total=args.num_samples)):
            if i >= args.num_samples: break

            dataset = TestDataset(item["image_path"], item["prompt"], res_num=1)
            batch = next(iter(DataLoader(dataset, batch_size=1, num_workers=0)))
            for key in batch:
                if isinstance(batch[key], torch.Tensor): batch[key] = batch[key].to(device)

            z, c = model.get_input(batch, model.first_stage_key, bs=1)
            cond = {"c_concat": c["c_concat"], "c_crossattn": c["c_crossattn"]}
            
            for t_val in [100, 500, 900]:
                t = torch.tensor([t_val], device=device).long()
                captured_activations.clear()
                _ = model.apply_model(z, t, cond, use_no_grad=False)

                for name in skippable_layer_names:
                    if name in captured_activations:
                        h_k = captured_activations[name]
                        h, w = h_k.shape[-2], h_k.shape[-1]
                        W_f = make_frequency_weight(h, w, args.band, device)
                        F_h = dct_2d(h_k)
                        freq_energy = torch.mean((F_h * W_f) ** 2).item()

                        layer_stats[name]["scores"].append(freq_energy)
                        layer_stats[name]["count"] += 1

    for handle in hook_handles: handle.remove()

    final_scores = []
    for name, stats in layer_stats.items():
        if stats["count"] > 0:
            scores = stats["scores"]
            final_scores.append({
                "name": name, 
                "score": np.mean(scores),
                "std_dev": np.std(scores) if len(scores) > 1 else 0
            })

    if not final_scores: print("\n[ERROR] No layer scores calculated for diffusion_model."); return
        
    final_scores.sort(key=lambda x: x["score"])
    print(f"\n--- Analysis Results for diffusion_model ---")
    print(f"\n{'Layer Name':<40} | {'Frequency Energy (Score)':<25} | {'Std Dev':<20}")
    print("-" * 90)
    for item in final_scores:
        print(f"{item['name']:<40} | {item['score']:<25.6e} | {item['std_dev']:<20.6e}")

    print("\n--- Generating Static Global Policy for diffusion_model ---")
    print(f"Using percentile {args.percentile} for skip threshold.") 
    
    policy_data = {"control_model": {},"diffusion_model": {}}
    all_scores = [s['score'] for s in final_scores]
    all_stds = [s['std_dev'] for s in final_scores]
    

    skip_threshold_mean = np.percentile(all_scores, args.percentile)
    run_threshold_mean = np.percentile(all_scores, 80) 
    std_threshold = np.percentile(all_stds, 50)

    for item in final_scores:
        score, std_dev, layer_name = item['score'], item['std_dev'], item['name']
        
        if score < skip_threshold_mean and std_dev < std_threshold:
            policy = "ALWAYS_SKIP"
        elif score > run_threshold_mean and std_dev < std_threshold:
            policy = "ALWAYS_RUN"
        else:
            policy = "DYNAMIC_DECISION"
            
        policy_data["diffusion_model"][layer_name] = policy


    output_path = args.output_filename if args.output_filename else f"policy_{args.band}.json"
    existing_policy = {}
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r') as f: existing_policy = json.load(f)
            print(f"Found existing policy file at '{output_path}'. Merging results.")
        except json.JSONDecodeError:
            print(f"Warning: Corrupted policy file '{output_path}'. Overwriting.")
            existing_policy = {"control_model": {}, "diffusion_model": {}}
    else:
        print(f"No existing policy file found. Creating new file at '{output_path}'.")
        existing_policy = {"control_model": {}, "diffusion_model": {}}

    if policy_data.get("diffusion_model"):
        existing_policy["diffusion_model"] = policy_data["diffusion_model"]
        print("Updated 'diffusion_model' section.")

    with open(output_path, 'w') as f:
        json.dump(existing_policy, f, indent=4)
    print(f"\n[SUCCESS] Merged static policy saved to '{output_path}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze diffusion_model layer importance via frequency energy.")
    parser.add_argument('--band', type=str, required=True, choices=['mini_pass', 'low_pass', 'mid_pass', 'high_pass'], help='The frequency band of the specialist model.')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to the model checkpoint.')
    parser.add_argument('--config', type=str, default='configs/model_config.yaml', help='Model config path.')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory with test images and prompts.')
    parser.add_argument('--num_samples', type=int, default=50, help='Number of samples to analyze.')
    parser.add_argument('--percentile', type=int, default=30, help='Bottom percentile for skiplist.')
    parser.add_argument('--output_filename', type=str, default=None, help="Optional: Specify a custom output filename for the policy JSON.")
      
    args = parser.parse_args()
    main(args)