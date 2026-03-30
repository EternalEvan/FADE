import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from omegaconf import OmegaConf
import imageio
from torchvision import transforms

from diffusion_core.guiders.guidance_editing_cogvideo import GuidanceEditing

from diffusion_core.utils import use_deterministic

from diffusers import CogVideoXPipeline

import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path", type=str, required=True, help="Path of the pretrained model"
    )
    parser.add_argument(
        "--init_prompt", type=str, required=True, help="Original video scene description"
    )
    parser.add_argument(
        "--edit_prompt", type=str, required=True, help="Target video scene description"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Compute device to use"
    )
    parser.add_argument(
        "--dtype", type=str, default="bf16", choices=["bf16", "fp16"], help="Dtype of the model"
    )
    parser.add_argument(
        "--input_video_path", type=str, required=True, help="Path to the input video file"
    )
    parser.add_argument(
        "--latent_trajectory_path", type=str, required=True, help="Path to the latent trajectory from inversion"
    )
    parser.add_argument(
        "--mask_path", type=str, required=True, help="Path to the generated mask tensor"
    )
    parser.add_argument(
        "--config_path", type=str, default="configs/bear.yaml", help="Path to the YAML configuration file for editing parameters"
    )
    parser.add_argument(
        "--fps", type=int, default=4, help="Frames per second of the input video"
    )
    parser.add_argument(
        "--output_video_path", type=str, required=True, help="Path to save the final edited video"
    )
    parser.add_argument(
        "--num_inference_steps", type=int, default=50, help="Number of diffusion steps for generation"
    )
    parser.add_argument(
        "--local_blend_end_step", type=int, default=35, help="Step at which to stop applying local blending"
    )

    args = parser.parse_args()  

    args.dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    
    return args
    
def main(args):
    
    use_deterministic()

    pipe = CogVideoXPipeline.from_pretrained(args.model_path, torch_dtype=args.dtype).to(args.device)
    pipe.scheduler.set_timesteps(num_inference_steps=args.num_inference_steps, device=args.device)
    
    
    config = OmegaConf.load(args.config_path)
    guidance = GuidanceEditing(pipe, config, args, lb=None)
    guidance(None, args.init_prompt, args.edit_prompt, verbose=True)

if __name__=="__main__":
    args = parse_args()
    main(args)
