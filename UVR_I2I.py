import os
import torch
from diffusers import FluxKontextPipeline
import torch
from PIL import Image
import math
from accelerate import init_empty_weights
from config import *
import importlib, types
from importlib import reload

pipe = FluxKontextPipeline.from_pretrained(
    Kontext_model_path,
    torch_dtype=torch.bfloat16,
    device_map="balanced", 
)
reload_fluxpipeline(pipe)
reload_kon_transformer(pipe)
prompt = "wear glasses"
unsafe_img_path = "unsafe.jpg"
image = Image.open(unsafe_img_path).resize((1024,1024))
pipes_out = pipe(
    image=image,
    prompt=prompt,
    save_anchor_flag = False,
    num_inference_steps=28,
    perturb_flag=True,
)
image = pipes_out.images[0]
image.save("flux.jpg")