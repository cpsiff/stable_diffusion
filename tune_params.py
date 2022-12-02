import torch
from torch import autocast
from PIL import Image
from SSIM_PIL import compare_ssim
import os
import yaml
import cv2
import numpy as np
import time
import io
import random

from diffusers import StableDiffusionImg2ImgPipeline

SOURCE_DIR = "cropped_images"
STRENGTHS = [0.001, 0.005, 0.01]
GUIDANCE_SCALES = [0.001, 0.005, 0.01]
SEED = 100
OUTPUT_DIR = "output"
NUM_COMBOS = 100

PROMPTS = [
    "4K, high definition, crisp desktop background, flickr picture of the day, pic of the day, Canon DSLR",
    "4K, high definition, crisp",
    "high definition deblurred denoised",
    "gradient, smooth high quality image, CGI render"
    "high quality, high resolution image"
    ""
]

def pil_to_cv2(pil_img):
    """Convert a PIL format image to CV2 format
    """
    arr = np.array(pil_img) 
    # Convert RGB to BGR 
    return(arr[:, :, ::-1].copy())


def quantize(img, colors=16):
    """Take PIL image as input and return quantized one
    """

    return img.quantize(colors=colors).convert("RGB")

def convert_to_jpg(img, qual=10):
    with io.BytesIO() as output:
        img.save(output, format="jpeg", quality=qual, subsampling=0)
        contents = output.getvalue()

    converted_img = Image.open(io.BytesIO(contents))
    return converted_img

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

    save_dir = os.path.join(OUTPUT_DIR, time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(save_dir)

    i = 0

    for _ in range(NUM_COMBOS):
        prompt = random.choice(PROMPTS)
        strength = random.choice(STRENGTHS)
        guidance_scale = random.choice(GUIDANCE_SCALES)

        for img_name in os.listdir(os.path.join(SOURCE_DIR)):
            init_img = Image.open(os.path.join(SOURCE_DIR, img_name))
            tf_img = transform_fn(init_img)

            with autocast("cuda"):
                result = pipe(
                    prompt=prompt,
                    init_image=tf_img,
                    strength=strength,
                    guidance_scale=guidance_scale,
                    generator=generator
                )
                
                is_nsfw = result.nsfw_content_detected
                image = result.images[0]
                save_name = f"{save_dir}/{i}.png"

                info = {
                    img_name: {
                        "prompt": prompt,
                        "guidance_scale": guidance_scale,
                        "strength": strength,
                        "seed": SEED,
                        "SSIM": compare_ssim(init_img, image),
                        "PSNR": cv2.PSNR(pil_to_cv2(init_img), pil_to_cv2(image)),
                        "L2": float(np.linalg.norm(np.asarray(init_img) - np.asarray(image))),
                        "is_nsfw": is_nsfw
                    }
                }
                with open(f"{save_dir}/index.yaml", "a+") as f:
                    yaml.dump(info, f)
                    f.write("\n")

                i += 1
                print(save_name)
                image.save(save_name)

if __name__ == "__main__":
    main()
