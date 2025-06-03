from src.pipeline_pe_clone import FluxPipeline
import torch
from PIL import Image

#pretrained_model_name_or_path = "black-forest-labs/FLUX.1-dev"
pretrained_model_name_or_path = "autodl-tmp/FLUX.1-dev"
pipeline = FluxPipeline.from_pretrained(
    pretrained_model_name_or_path,
    torch_dtype=torch.bfloat16,
).to('cuda')

#pipeline.load_lora_weights("nicolaus-huang/PhotoDoodle", weight_name="pretrain.safetensors")
pipeline.load_lora_weights("autodl-tmp/1024_model/first_stage_1024.safetensors")
pipeline.fuse_lora()
pipeline.unload_lora_weights()

#pipeline.save_pretrained("PhotoDoodle_Pretrain")
pipeline.save_pretrained("autodl-tmp/pretrain")