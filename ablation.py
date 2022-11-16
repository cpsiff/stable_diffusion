import torch
from torch import autocast
from tqdm.auto import tqdm
from PIL import Image
import numpy as np
import os

from diffusers import StableDiffusionImg2ImgPipeline

RESIZE = 512

device = "cuda"
model_path = "CompVis/stable-diffusion-v1-4"

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_path,
    revision="fp16", 
    torch_dtype=torch.float16,
    use_auth_token=True
)
pipe = pipe.to(device)

prompt = "A fantasy landscape, trending on artstation"

generator = torch.Generator(device=device).manual_seed(1024)

for fold_name in os.listdir("corrupted_imgs"):
    for img_name in os.listdir(os.path.join("corrupted_imgs", fold_name)):
        init_img = Image.open(os.path.join("corrupted_imgs", fold_name, img_name))
        
        if RESIZE:
            if init_img.size[0] > init_img.size[1]:
                init_img = init_img.resize((RESIZE, int(init_img.size[1]*(RESIZE/init_img.size[0]))))
            else:
                init_img = init_img.resize((int(init_img.size[0]*(RESIZE/init_img.size[1])), RESIZE))

        for strength in np.linspace(0.1, 0.9, 3):
            for guidance_scale in np.linspace(1, 9, 3):
                with autocast("cuda"):
                    image = pipe(prompt=prompt, init_image=init_img, strength=strength, guidance_scale=guidance_scale, generator=generator).images[0]
                    # image = init_img
                    save_name = f"output/{fold_name}_{img_name.split('.')[0]}_s{str(strength).replace('.', 'dot')}_gs{str(guidance_scale).replace('.', 'dot')}.png"
                    print(save_name)
                    image.save(save_name)