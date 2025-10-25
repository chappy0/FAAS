

import torch
import numpy as np
import argparse
import time
from PIL import Image
from omegaconf import OmegaConf
import os
from tqdm import tqdm
import traceback
import json

from ldm.util import instantiate_from_config
from fcdiffusion.router import PromptComplexityRouter
from torch.utils.data import DataLoader
from fcdiffusion.dataset import TestDataset
from fcdiffusion.fcdiffusion_highpass import FCDiffusion 

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

def load_static_policy(filepath):
    if not filepath or not os.path.exists(filepath):
        print(f"Warning: Policy file '{filepath}' not found. No static policies loaded.")
        return {}
    try:
        with open(filepath, 'r') as f:
            policy = json.load(f)
        print(f"Successfully loaded static policy for {len(policy.get('control_model', {}))} control layers and {len(policy.get('diffusion_model', {}))} diffusion layers.")
        return policy
    except Exception as e:
        print(f"Error loading policy file {filepath}: {e}")
        return {}

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    MODEL_CONFIGS = {
        'high_pass': {
            'ckpt_path': 'path/your/high pass checkpoint',
            'policy_file': 'policy_high_pass.json'
        },
        'low_pass': {
            'ckpt_path': 'path/your/low pass checkpoint',
            'policy_file': 'policy_low_pass.json'
        },
        'mid_pass': {
            'ckpt_path': 'path/your/mid pass checkpoint',
            'policy_file': 'policy_mid_pass.json'
        },
        'mini_pass': {
            'ckpt_path': 'path/your/mini pass checkpoint',
            'policy_file': 'policy_mini_pass.json'
        }
    }


    if args.analyze_only:
        print("\n--- RUNNING IN ANALYSIS-ONLY MODE ---")
        
        config = OmegaConf.load(args.fcd_config)
        print("Loading FCDiffusion model for text encoder...")
        fcd_model = instantiate_from_config(config.model).to(device)
        temp_ckpt_path = list(MODEL_CONFIGS.values())[0]['ckpt_path']
        print(f"Loading a dummy checkpoint for text encoder from: {temp_ckpt_path}")
        fcd_model.load_and_migrate_checkpoint(temp_ckpt_path)
        fcd_model.eval()

        print(f"Loading universal Router from: {args.router_ckpt}")
        router = PromptComplexityRouter(embedding_dim=fcd_model.cond_stage_model.model.transformer.width).to(device)
        router.load_state_dict(torch.load(args.router_ckpt, map_location=device))
        router.eval()

        image_text_pairs = get_image_text_pairs(args.input_dir)
        scores = []
        with torch.no_grad():
            for item in tqdm(image_text_pairs, desc="Analyzing prompt scores"):
                prompt = item['prompt']
                prompt_embedding = fcd_model.get_learned_conditioning([prompt]).to(device)
                complexity_logit = router(prompt_embedding)
                complexity_score = torch.sigmoid(complexity_logit).item()
                scores.append(complexity_score)
        
        if scores:
            scores = np.array(scores)
            print("\n--- Complexity Score Distribution Analysis ---")
            print(f"Total prompts analyzed: {len(scores)}")
            print(f"Min score:    {np.min(scores):.4f}")
            print(f"Max score:    {np.max(scores):.4f}")
            print(f"Mean score:   {np.mean(scores):.4f}")
            print(f"Median score (50th percentile): {np.median(scores):.4f}")
            print(f"25th percentile: {np.percentile(scores, 25):.4f}")
            print(f"75th percentile: {np.percentile(scores, 75):.4f}")
            print("\nRECOMMENDATION: Use the 'Median score' as your new --threshold for a balanced split.")
        else:
            print("No data found to analyze.")
        
        return 
    print(f"Selected Inference Mode: {args.inference_mode.upper()}")
    
    selected_config = MODEL_CONFIGS[args.control_mode]
    config = OmegaConf.load(args.fcd_config)
    config.model.params.control_mode = args.control_mode
    
    print(f"Instantiating FCDiffusion model from {args.fcd_config}...")
    fcd_model = instantiate_from_config(config.model)
    fcd_model.load_and_migrate_checkpoint(selected_config['ckpt_path'])
    fcd_model.to(device).eval()

    router = None
    if args.inference_mode != 'full':
        print(f"Loading universal Router from: {args.router_ckpt}")
        router = PromptComplexityRouter(embedding_dim=fcd_model.cond_stage_model.model.transformer.width).to(device).eval()
        router.load_state_dict(torch.load(args.router_ckpt, map_location=device))

    if args.policy_file:
        print(f"Loading policy from specified file: {args.policy_file}")
        static_policy = load_static_policy(args.policy_file)
    else:
        static_policy = load_static_policy(selected_config['policy_file'])

    control_skiplist_static = {name for name, policy in static_policy.get('control_model', {}).items() if policy == 'ALWAYS_SKIP'}
    diffusion_skiplist_static = {name for name, policy in static_policy.get('diffusion_model', {}).items() if policy == 'ALWAYS_SKIP'}
    
    image_text_pairs = get_image_text_pairs(args.input_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    
    total_time = 0
    cache_manager = None
    if args.use_deepcache:
        from fcdiffusion.fcdiffusion import DeepCacheManager
        cache_manager = DeepCacheManager(
            cache_start_step=args.deepcache_start_step,
            cache_interval=args.deepcache_interval
        )

    for i, item in enumerate(tqdm(image_text_pairs, desc=f"Inference for '{args.control_mode}'")):
        image_path, prompt = item['image_path'], item['prompt']
        print(f"\n--- Processing {i+1}/{len(image_text_pairs)}: {os.path.basename(image_path)} ---")

        try:
            dataset = TestDataset(image_path, prompt, res_num=1)
            batch = next(iter(DataLoader(dataset, batch_size=1, num_workers=0)))

            with torch.no_grad():
                active_control_skiplist, active_diffusion_skiplist = None, None
                active_policy = None
                current_inference_mode = args.inference_mode
                top_level_mode = "" 

                if args.inference_mode == 'full':
                    print("Inference mode is 'full', running with full computation.")
                    top_level_mode = "full" 
                else:
                    prompt_embedding = fcd_model.get_learned_conditioning([prompt]).to(device)
                    complexity_logit = router(prompt_embedding)
                    complexity_score = torch.sigmoid(complexity_logit).item()
                    
                    if complexity_score < args.threshold:
                        top_level_mode = "SIMPLE"
                        if args.inference_mode == 'static':
                            active_control_skiplist = control_skiplist_static
                            active_diffusion_skiplist = diffusion_skiplist_static
                        else: 
                            active_policy = static_policy
                    else:
                        top_level_mode = "COMPLEX"
                        current_inference_mode = 'full' 
                    
                    print(f"Prompt complexity score: {complexity_score:.4f} -> Top-level path: {top_level_mode}")
                    if top_level_mode == 'SIMPLE':
                        print(f"  -> Using sub-mode: {args.inference_mode}")

                start_time = time.time()
                log = fcd_model.log_images(
                    batch, 
                    ddim_steps=args.ddim_steps,
                    unconditional_guidance_scale=args.guidance_scale,
                    skiplist_control=active_control_skiplist,
                    skiplist_diffusion=active_diffusion_skiplist,
                    static_policy=active_policy,
                    inference_mode=current_inference_mode,
                    cache_manager=cache_manager, 
                    N=1,
                    dynamic_threshold={"percentile": 0.995},
                )
                inference_time = time.time() - start_time
                total_time += inference_time
                print(f"Sampling finished in {inference_time:.2f} seconds.")

                sample_tensor = log["samples"].squeeze(0)
                sample_tensor = (sample_tensor + 1.0) / 2.0
                sample_tensor = torch.clamp(sample_tensor, 0.0, 1.0)
                sample_numpy = (sample_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                pil_image = Image.fromarray(sample_numpy)
                
                output_filename = f"{args.inference_mode}_{top_level_mode.lower()}_{os.path.basename(image_path)}"
                output_path = os.path.join(args.output_dir, output_filename)
                
                pil_image.save(output_path)
                print(f"Output image saved to: {output_path}")

        except Exception as e:
            print(f"!!!!!! ERROR processing {image_path}: {e} !!!!!!")
            traceback.print_exc()
            continue

    if len(image_text_pairs) > 0:
        print(f"\n--- Batch processing finished ---")
        print(f"Average inference time: {total_time / len(image_text_pairs):.2f} seconds per image.")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run BATCH dynamic inference with FCDiffusion.")
    parser.add_argument('--analyze_only', action='store_true', 
                        help="Run in analysis-only mode to find a good threshold. Does not generate images.")
    
    parser.add_argument('--inference_mode', type=str, default='static', choices=['static', 'adaptive','full'], 
                        help="Choose inference sub-mode for the SIMPLE path: 'static' uses a fixed skiplist, 'adaptive' uses dynamic probes.")
    
    parser.add_argument('--input_dir', type=str, required=True, help='Path to the directory containing image-text pairs.')
    parser.add_argument('--output_dir', type=str, help='Path to save images. Not required for --analyze_only.')
    parser.add_argument('--router_ckpt', type=str, default='models/prompt_router.ckpt', help='Path to the trained Router checkpoint.')
    parser.add_argument('--control_mode', type=str, required=True, choices=['mini_pass', 'low_pass', 'mid_pass', 'high_pass'], help='The FCDiffusion specialist task mode to use.')
    parser.add_argument('--fcd_config', type=str, default='configs/model_config.yaml', help='Path to the base FCDiffusion model config file.')
    parser.add_argument('--threshold', type=float, default=0.5, help='Complexity score threshold. Prompts below this use the SIMPLE path.')
    parser.add_argument('--ddim_steps', type=int, default=50, help='Number of DDIM sampling steps.')
    parser.add_argument('--guidance_scale', type=float, default=9.0, help='Unconditional guidance scale.')
    
    parser.add_argument('--use_deepcache', action='store_true', help="Enable DeepCache acceleration.")
    parser.add_argument('--deepcache_start_step', type=int, default=500, help="Timestep below which to start caching for DeepCache.")
    parser.add_argument('--deepcache_interval', type=int, default=3, help="The interval for updating the cache in DeepCache.")
    parser.add_argument('--policy_file', type=str, default="policy_high_pass_aftersense.json", help='Path to a specific policy JSON file to use, overriding the default from MODEL_CONFIGS.')
    
    args = parser.parse_args()

    if not args.analyze_only and not args.output_dir:
        parser.error("--output_dir is required unless running with --analyze_only.")

    main(args)