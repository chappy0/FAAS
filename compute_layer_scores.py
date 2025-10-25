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
from ldm.modules.diffusionmodules.openaimodel import Downsample 


control_outputs = []

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

def get_control_outputs_hook(model, input, output):
    global control_outputs
    control_outputs = [[o[0].detach(), o[1].detach()] for o in output]

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = OmegaConf.load(args.config)
    config.model.params.control_mode = args.band
    model = instantiate_from_config(config.model)
    model.load_and_migrate_checkpoint(args.ckpt)
    model.to(device).eval()
    control_model = model.control_model

    hook_handle = control_model.register_forward_hook(get_control_outputs_hook)
    print("Hook registered on control_model. Analyzing it in isolation.")

    image_text_pairs = get_image_text_pairs(args.data_dir)
    if not image_text_pairs:
        print("No image-text pairs found. Exiting."); return
        
    layer_stats = defaultdict(lambda: {"count": 0, "scores": []})
    
    print(f"Analyzing {args.num_samples} samples for control_model...")
    with torch.no_grad():
        for i, item in enumerate(tqdm(image_text_pairs, total=args.num_samples)):
            if i >= args.num_samples: break
            
            dataset = TestDataset(item["image_path"], item["prompt"], res_num=1)
            batch = next(iter(DataLoader(dataset, batch_size=1, num_workers=0)))
            
            for key in batch:
                if isinstance(batch[key], torch.Tensor): batch[key] = batch[key].to(device)

            z, c = model.get_input(batch, model.first_stage_key, bs=1)
            
            for t_val in [100, 500, 900]:
                t = torch.tensor([t_val], device=device).long()
                global control_outputs; control_outputs = []

                _ = control_model(x=z, hint=c["c_concat"][0], timesteps=t, context=c["c_crossattn"][0])
                
                if not control_outputs: continue

                for k, pair in enumerate(control_outputs):
                    block_name = f"input_blocks.{k}" if k < len(control_model.input_blocks) else "middle_block"
                    
                    add_signal, mul_signal = pair
                    
                    a_k_energy = torch.mean(add_signal**2).item()
                    m_k_energy = torch.mean(mul_signal**2).item()
                    
                    t_weight = 1.0 if t_val < 400 else 0.5
                    score = t_weight * (args.alpha * a_k_energy + args.beta * m_k_energy)

                    layer_stats[block_name]["scores"].append(score)
                    layer_stats[block_name]["count"] += 1

    hook_handle.remove()

    final_scores = []
    for name, stats in layer_stats.items():
        if stats["count"] > 0:
            scores = stats["scores"]
            final_scores.append({
                "name": name, 
                "score": np.mean(scores),
                "std_dev": np.std(scores) if len(scores) > 1 else 0
            })
            
    print("\n--- Performing Global Normalization (on non-critical layers) ---")
    
    critical_layers = ["input_blocks.1", "input_blocks.2"]
    
    non_critical_scores = []
    for item in final_scores:
        if item["name"] not in critical_layers:
            non_critical_scores.append(item["score"])
    
    if not non_critical_scores:
        print("[Warning] No non-critical layers found to normalize. Skipping normalization.")
    else:

        global_mean = np.mean(non_critical_scores)
        global_std = np.std(non_critical_scores) + 1e-8 
        
        print(f"Global Stats (Non-Critical): Mean={global_mean:.6e}, Std={global_std:.6e}")

        for item in final_scores:
            if item["name"] not in critical_layers:
                item["score_norm"] = (item["score"] - global_mean) / global_std


    final_scores.sort(key=lambda x: x.get("score_norm", float('inf')))
        
    print(f"\n--- Analysis Results for control_model (Normalized) ---")
    print(f"\n{'Block Name':<40} | {'Normalized Score':<25} | {'Original Score':<25}")
    print("-" * 100)
    for item in final_scores:
        if item["name"] in critical_layers:
             norm_score_str = "N/A (Critical Layer)"
        else:
             norm_score_str = f"{item.get('score_norm', 'N/A'):.6f}" if 'score_norm' in item else "N/A"
        print(f"{item['name']:<40} | {norm_score_str:<25} | {item['score']:<25.6e}")


    print("\n--- Generating Static Global Policy for control_model ---")
    policy_data = {"control_model": {}, "diffusion_model": {}}
    
    if not final_scores: 
        print("\n[ERROR] No scores were calculated, cannot generate policy. Exiting.")
        return

    
    for item in final_scores:
        block_name = item["name"]
        score_norm = item.get("score_norm", 0.0) 
        
        if any(cl == block_name for cl in critical_layers):
            policy = "ALWAYS_RUN"
        elif score_norm < args.skip_threshold:
            policy = "ALWAYS_SKIP"
        elif score_norm > args.run_threshold:
            policy = "ALWAYS_RUN"
        else:
            policy = "DYNAMIC_DECISION"
        
        if block_name == "middle_block":
            block_module = control_model.middle_block
        else:
            block_idx = int(block_name.split('.')[1])
            block_module = control_model.input_blocks[block_idx]

        if hasattr(block_module, '__iter__'):
            for sub_idx, sub_layer in enumerate(block_module):
                if isinstance(sub_layer, ConditionallySkipableBlock):
                    granular_name = f"{block_name}.{sub_idx}"
                    policy_data["control_model"][granular_name] = policy
        
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

    if policy_data.get("control_model"):
        existing_policy["control_model"] = policy_data["control_model"]
        print("Updated 'control_model' section.")

    with open(output_path, 'w') as f:
        json.dump(existing_policy, f, indent=4)
    print(f"\n[SUCCESS] Merged static policy saved to '{output_path}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze control_model layer importance using weighted signal energy.")
    parser.add_argument('--band', type=str, required=True, choices=['mini_pass', 'low_pass', 'mid_pass', 'high_pass'])
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--config', type=str, default='configs/model_config.yaml')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--skip_threshold', type=float, default=-0.5, help="Normalized score below which a layer is marked ALWAYS_SKIP.")
    # parser.add_argument('--run_threshold', type=float, default=0.5, help="Normalized score above which a layer is marked ALWAYS_RUN.")
    parser.add_argument('--run_threshold', type=float, default=None, help="Normalized score above which a layer is marked ALWAYS_RUN. Defaults to -skip_threshold.")
    parser.add_argument('--alpha', type=float, default=1.0, help="Weight for the add_signal energy.")
    parser.add_argument('--beta', type=float, default=0.1, help="Weight for the mul_signal energy.")
    parser.add_argument('--output_filename', type=str, default=None, help="Optional: Specify a custom output filename for the policy JSON.")
    
    args = parser.parse_args()
    if args.run_threshold is None:
        args.run_threshold = -args.skip_threshold
    main(args)