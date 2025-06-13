import argparse
from src.pipeline_pe_clone_multisample import FluxPipeline
import torch
from PIL import Image
import numpy as np
import random
import os
import logging

def parse_args():
    parser = argparse.ArgumentParser(description='FLUX image generation with LoRA')
    parser.add_argument('--model_path', type=str, 
                        default="path/to/FLUX.1-dev",
                        help='Path to base model')
    parser.add_argument('--data_path', type=str,
                        default="sample/prompt",
                        help='Input prompt path')
    parser.add_argument('--data_condition_path', type=str,
                        default="sample/condition",
                        help='Input data_condition path')
    parser.add_argument('--output_path', type=str,
                        default="sample/output",
                        help='Output image path')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--height', type=int, default=1024)
    parser.add_argument('--width', type=int, default=1024)
    parser.add_argument('--guidance_scale', type=float, default=3.5)
    parser.add_argument('--num_steps', type=int, default=20,
                        help='Number of inference steps')
    return parser.parse_args()

def main():
    args = parse_args()
    pipeline = FluxPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
    ).to('cuda')
    seed = 42
    torch.manual_seed(seed)
    # Load and fuse pretrain LoRA weights
    pipeline.load_lora_weights("path/to/pretrain lora")
    pipeline.fuse_lora()
    pipeline.unload_lora_weights()
    # Load fine-tune LoRA
    pipeline.load_lora_weights("path/to/fine-tuning lora")
    def process_condition_image(image_path: str) -> Image.Image:
        condition_image = Image.open(image_path).convert("RGB")
        return condition_image
    data_folder_path = args.data_path
    data_condition_folder_path = args.data_condition_path
    output_base_path = args.output_path

    for txt_filename in os.listdir(data_folder_path):
        if txt_filename.endswith('.txt') and txt_filename[0].isdigit():
            txt_file_path = os.path.join(data_folder_path, txt_filename)
            sample_id = txt_filename.split('.')[0]

            with open(txt_file_path, 'r') as file:
                caption = file.read()


            condition_image_path = os.path.join(data_condition_folder_path, f"{sample_id}-text-wb.png")
            if not os.path.exists(condition_image_path):
                print(f"Image file not found, skipping current iteration: {condition_image_path}")
                continue

            condition_image = process_condition_image(condition_image_path)
            position_file = os.path.join(data_condition_folder_path, f"{sample_id}-position.txt")
            if not os.path.exists(position_file):
                print(f"Text control file not found, skipping current iteration.: {position_file}")
                continue

            output_path = os.path.join(output_base_path, f"{sample_id}.png")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            result = pipeline(
                prompt=caption,
                condition_image=condition_image,
                height=args.height,
                width=args.width,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_steps,
                position_file=position_file,
                max_sequence_length=512
            ).images[0]
            result.save(output_path)


if __name__ == "__main__":
    main()
