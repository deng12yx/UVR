import os
import torch
from diffusers import FluxPipeline
import torch
from PIL import Image
import math
from accelerate import init_empty_weights
from config import *
import importlib, types
from importlib import reload

pipe = FluxPipeline.from_pretrained(
    Dev_model_path,
    torch_dtype=torch.bfloat16,
    device_map="balanced", 
)
reload_fluxpipeline(pipe)
reload_dev_transformer(pipe)
prompt = "a picture of body shot outdoors under natural light ,<Possession> ,loneliness ,Wood carving ,Art Deco ,feminine dark fantasy"
pipes_out = pipe(
    prompt=prompt,
    save_anchor_flag = False,
    num_inference_steps=28,
    perturb_flag=True,
)
