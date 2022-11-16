import torch
from torch import autocast
from tqdm.auto import tqdm
from PIL import Image
import numpy as np
import os
import yaml

from diffusers import StableDiffusionImg2ImgPipeline

FOLDS = "JPEG", "PALETTE"
RESIZE = 512
STRENGTHS = [0.1]
GUIDANCE_SCALES = [0.5, 1]
SEED = 100
OUTPUT_DIR = "output"

PROMPTS = [
    "4K, high definition, crisp desktop background",
    "4K, high definition, crisp",
    "high definition deblurred denoised",
]

device = "cuda"
model_path = "CompVis/stable-diffusion-v1-4"

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_path,
    revision="fp16", 
    torch_dtype=torch.float16,
    use_auth_token=True
)
pipe = pipe.to(device)

generator = torch.Generator(device=device).manual_seed(SEED)

index = {} #store an index of prompts used for different file names

for fold_name in FOLDS:
    for img_name in os.listdir(os.path.join("corrupted_imgs", fold_name)):
        init_img = Image.open(os.path.join("corrupted_imgs", fold_name, img_name))
        
        if RESIZE:
            if init_img.size[0] > init_img.size[1]:
                init_img = init_img.resize((RESIZE, int(init_img.size[1]*(RESIZE/init_img.size[0]))))
            else:
                init_img = init_img.resize((int(init_img.size[0]*(RESIZE/init_img.size[1])), RESIZE))

        for strength in STRENGTHS:
            for guidance_scale in GUIDANCE_SCALES:
                for prompt, i in zip(PROMPTS, range(len(PROMPTS))):  
                    with autocast("cuda"):
                        image = pipe(
                            prompt=prompt,
                            init_image=init_img,
                            strength=strength,
                            guidance_scale=guidance_scale,
                            generator=generator
                        ).images[0]
                        # image = init_img
                        save_name = f"{OUTPUT_DIR}/{fold_name}_{img_name.split('.')[0]}_s{str(strength).replace('.', 'dot')}_gs{str(guidance_scale).replace('.', 'dot')}_{i}.png"
                        index[save_name] = {
                            "prompt": prompt,
                            "guidance_scale": guidance_scale,
                            "strength": strength,
                            "fold_name": fold_name,
                            "img_name": img_name,
                            "seed": SEED,
                            "resize": RESIZE
                        }
                        print(save_name)
                        image.save(save_name)

with open(f"{OUTPUT_DIR}/index.yaml") as f:
    yaml.dump(index, f)