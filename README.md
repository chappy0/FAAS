FAAS: Frequency-Aware Adaptive Skipping for Accelerated Diffusion Model Inference

 <!-- TODO: Replace <COLOR> and <https://www.google.com/search?q=YOUR_ARXIV_LINK_HERE> -->

This repository contains the official implementation for the paper: "Frequency-Aware Adaptive Skipping for Accelerated Diffusion Model Inference".

FAAS is a training-free framework designed to accelerate the inference speed of conditional diffusion models, specifically tailored for the frequency-controlled FCDiffusion architecture for text-guided image-to-image (I2I) translation. It dynamically skips redundant computational blocks within the U-Net backbone based on a combination of pre-computed frequency-aware policies and lightweight runtime probes, achieving significant speedups while maintaining high visual fidelity.

Key Features

Training-Free: Directly applicable to pre-trained FCDiffusion checkpoints (based on Stable Diffusion v2.1) without retraining.

Frequency-Aware Policies: Utilizes offline analysis scripts (compute_layer_scores.py, compute_layer_scores_diffusion_freq.py) to generate static policy maps (policy_*.json) based on layer importance in specific frequency bands.

Adaptive Skipping Logic: The core dynamic_inference.py script integrates:

A Prompt Complexity Router (prompt_router.ckpt) to determine if a prompt requires full computation (COMPLEX path) or can utilize skipping (SIMPLE path).

Static Mode: In the SIMPLE path, uses the pre-computed static policy to always skip designated low-importance layers.

Adaptive Mode: In the SIMPLE path, uses the static policy as a prior, but dynamically probes layers marked as DYNAMIC_DECISION at runtime to decide whether to skip.

Orthogonal Acceleration: Compatible with DeepCache-style temporal caching (--use_deepcache) for cumulative speedup.

Setup

Clone Repository:


cd FAAS-Project


Create Conda Environment (Recommended):

conda create -n faas python=3.9 # Or your Python version
conda activate faas


Install Dependencies:

pip install torch torchvision toraudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118) # Adjust for your CUDA version
pip install -r requirements.txt


Ensure requirements.txt includes omegaconf, einops, transformers, pytorch-lightning, kornia, diffusers, scipy, matplotlib, etc. (Check imports in provided scripts).

Download Pre-trained Models & Policies:

FCDiffusion Checkpoints: Download the FCDiffusion checkpoints (.ckpt) for the frequency bands you intend to use (e.g., low_pass, high_pass). Place them in an accessible directory.

# Example structure:
# models/fcdiffusion_checkpoints/fcdiffusion_high_pass_checkpoint/epoch=X-step=Y.ckpt
# models/fcdiffusion_checkpoints/fcdiffusion_low_pass_checkpoint/epoch=X-step=Y.ckpt


(Code currently points to hardcoded paths like /home/apulis-dev/userdata/.... Update these in dynamic_inference.py or make them configurable.)

Prompt Router: Download the prompt_router.ckpt file and place it (e.g., in a models/ directory). Update the default path in dynamic_inference.py if needed.

# Example location: models/prompt_router.ckpt


Static Policy Maps: Pre-computed policy maps (policy_high_pass.json, policy_low_pass.json, etc.) should be available. Place them where the scripts expect them (e.g., project root or a policy_maps/ directory).

# Example locations: ./policy_high_pass_0.65_20.json, ./policy_low_pass.json


Stable Diffusion v2.1 Base: Ensure the underlying Stable Diffusion v2.1 weights (used by FCDiffusion) are accessible (often handled by the ldm library or require separate download). (Clarify this if needed).

Usage

The main script for inference is dynamic_inference.py.

1. Running Inference

Choose an inference mode (--inference_mode) and specify the FCDiffusion frequency band (--control_mode).

Core Arguments:

--input_dir: Path to the directory containing image (.png/.jpg) and corresponding prompt (.txt) pairs.

--output_dir: Path where generated images will be saved.

--control_mode: The FCDiffusion frequency band to use (e.g., low_pass, high_pass). This determines which checkpoint is loaded.

--router_ckpt: Path to the prompt_router.ckpt file.

--threshold: The complexity score threshold (default: 0.5). Prompts scoring below this use the SIMPLE path (static/adaptive skipping).

--inference_mode: Choose the strategy for the SIMPLE path:

static: Always skips layers marked ALWAYS_SKIP in the policy file.

adaptive: Uses the policy file, but dynamically probes layers marked DYNAMIC_DECISION.

full: Runs the full FCDiffusion model (no skipping, ignores Router/Policy). Note: The script currently overrides to 'full' for COMPLEX prompts.

--policy_file: (Optional) Path to a specific policy JSON file. If not provided, uses the default policy associated with the --control_mode.

--use_deepcache: (Optional) Enable DeepCache integration.

--ddim_steps, --guidance_scale: Standard DDIM parameters.

Example Commands:

Baseline (Full FCDiffusion):

python dynamic_inference.py \
    --input_dir ./datasets/test_samples \
    --output_dir ./outputs/full_results \
    --control_mode high_pass \
    --inference_mode full \
    --fcd_config configs/model_config.yaml \
    --ddim_steps 50 --guidance_scale 9.0


FAAS (Static Mode): Uses Router; skips based only on ALWAYS_SKIP in the policy for SIMPLE prompts.

python dynamic_inference.py \
    --input_dir ./datasets/test_samples \
    --output_dir ./outputs/static_results \
    --control_mode high_pass \
    --inference_mode static \
    --router_ckpt models/prompt_router.ckpt \
    --threshold 0.5 \
    --policy_file policy_high_pass_0.65_20.json \
    --fcd_config configs/model_config.yaml \
    --ddim_steps 50 --guidance_scale 9.0


FAAS (Adaptive Mode - Recommended): Uses Router; skips based on ALWAYS_SKIP and dynamically probes DYNAMIC_DECISION layers for SIMPLE prompts.

python dynamic_inference.py \
    --input_dir ./datasets/test_samples \
    --output_dir ./outputs/adaptive_results \
    --control_mode high_pass \
    --inference_mode adaptive \
    --router_ckpt models/prompt_router.ckpt \
    --threshold 0.5 \
    --policy_file policy_high_pass_0.65_20.json \
    --fcd_config configs/model_config.yaml \
    --ddim_steps 50 --guidance_scale 9.0


FAAS (Adaptive Mode with DeepCache):

python dynamic_inference.py \
    --input_dir ./datasets/test_samples \
    --output_dir ./outputs/adaptive_deepcache_results \
    --control_mode high_pass \
    --inference_mode adaptive \
    --router_ckpt models/prompt_router.ckpt \
    --threshold 0.5 \
    --policy_file policy_high_pass_0.65_20.json \
    --fcd_config configs/model_config.yaml \
    --ddim_steps 50 --guidance_scale 9.0 \
    --use_deepcache \
    --deepcache_start_step 500 --deepcache_interval 3 # Adjust params as needed


2. Analyzing Prompt Complexity Threshold (Optional)

To find a suitable --threshold value for your dataset, you can run dynamic_inference.py in analysis mode:

python dynamic_inference.py --analyze_only \
    --input_dir ./datasets/your_analysis_set \
    --router_ckpt models/prompt_router.ckpt \
    --control_mode high_pass # Need any valid control_mode to load text encoder
    --fcd_config configs/model_config.yaml


This will print statistics about the complexity scores (min, max, mean, median, percentiles) for the prompts in --input_dir. The median score is often a good starting point for --threshold.

3. Regenerating Policy Maps (Advanced)

The static policy maps (policy_*.json) are generated by analyzing layer importance offline. We provide pre-computed maps. If you need to regenerate them (e.g., using a different dataset or parameters):

For control_model (Signal Energy):

python compute_layer_scores.py \
    --band low_pass \
    --ckpt <path_to_low_pass_ckpt> \
    --config configs/model_config.yaml \
    --data_dir <path_to_analysis_dataset> \
    --num_samples 100 \
    --skip_threshold -0.5 --run_threshold 0.5 # Adjust thresholds
    --alpha 1.0 --beta 0.1 \
    --output_filename policy_low_pass.json


For diffusion_model (Frequency Energy):

python compute_layer_scores_diffusion_freq.py \
    --band low_pass \
    --ckpt <path_to_low_pass_ckpt> \
    --config configs/model_config.yaml \
    --data_dir <path_to_analysis_dataset> \
    --num_samples 50 \
    --percentile 30 # Adjust percentile for skipping
    --output_filename policy_low_pass.json


Note: These scripts merge results into the output JSON. Ensure parameters (--band, --ckpt, thresholds/percentiles, --output_filename) match your needs.




