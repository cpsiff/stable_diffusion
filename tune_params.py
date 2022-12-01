import torch
from torch import autocast
from PIL import Image
from SSIM_PIL import compare_ssim
import os
import yaml
import cv2
import numpy as np

from diffusers import StableDiffusionImg2ImgPipeline

SOURCE_DIR = "cropped_images"
STRENGTHS = [0.1]
GUIDANCE_SCALES = [0.1]
SEED = 100
OUTPUT_DIR = "output"

PROMPTS = [
    "4K, high definition, crisp desktop background, flickr picture of the day, pic of the day, Canon DSLR",
#    "4K, high definition, crisp",
#    "high definition deblurred denoised",
#    "fast shutter speed, 4K high definition, deblurred, denoised"
]

def pil_to_cv2(pil_img):
    """Convert a PIL format image to CV2 format
    """
    arr = np.array(pil_img) 
    # Convert RGB to BGR 
    return(arr[:, :, ::-1].copy())


def quantize(img, colors=12):
    """Take PIL image as input and return quantized one
    """

    return img.quantize(colors=colors).convert("RGB")

def main():
    ablate(quantize)


def ablate(transform_fn):
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

    i = 0
    for img_name in os.listdir(os.path.join(SOURCE_DIR)):
        init_img = Image.open(os.path.join(SOURCE_DIR, img_name))
        tf_img = transform_fn(init_img)
        for strength in STRENGTHS:
            for guidance_scale in GUIDANCE_SCALES:
                for prompt in PROMPTS:  
                    with autocast("cuda"):
                        image = pipe(
                            prompt=prompt,
                            init_image=tf_img,
                            strength=strength,
                            guidance_scale=guidance_scale,
                            generator=generator
                        ).images[0]
                        save_name = f"{OUTPUT_DIR}/{OUTPUT_DIR}_{i}.png"

                        index[save_name] = {
                            "prompt": prompt,
                            "guidance_scale": guidance_scale,
                            "strength": strength,
                            "img_name": img_name,
                            "seed": SEED,
                            "SSIM": compare_ssim(init_img, image),
                            # "PSNR": cv2.PSNR(pil_to_cv2(init_img), pil_to_cv2(image)),
                            "L2": np.linalg.norm(np.array(init_img) - np.array(image))
                        }
                        i += 1
                        print(save_name)
                        image.save(save_name)

    with open(f"{OUTPUT_DIR}/index.yaml", "w+") as f:
        yaml.dump(index, f)

if __name__ == "__main__":
    main()
