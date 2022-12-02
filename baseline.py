"""Get baseline stats by just transforming image and not applying any sort of improvement to it
"""

from PIL import Image
from SSIM_PIL import compare_ssim
import os
import yaml
import cv2
import numpy as np
import time
import io

SOURCE_DIR = "cropped_images"
OUTPUT_DIR = "baseline"

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

def convert_to_jpg(img, qual):
    with io.BytesIO() as output:
        img.save(output, format="jpeg", quality=qual, subsampling=0)
        contents = output.getvalue()

    converted_img = Image.open(io.BytesIO(contents))
    return converted_img

def main():
    get_baseline(quantize)


def get_baseline(transform_fn):
    save_dir = os.path.join(OUTPUT_DIR, time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(save_dir)

    i = 0

    for img_name in os.listdir(os.path.join(SOURCE_DIR)):
        init_img = Image.open(os.path.join(SOURCE_DIR, img_name))
        tf_img = transform_fn(init_img)

        save_name = f"{save_dir}/{i}.png"

        info = {
            img_name: {
                "prompt": "BASELINE",
                "seed": 0,
                "strength": 0.0,
                "guidance_scale": 0.0,
                "is_nsfw": [False],
                "SSIM": compare_ssim(init_img, tf_img),
                "PSNR": cv2.PSNR(pil_to_cv2(init_img), pil_to_cv2(tf_img)),
                "L2": float(np.linalg.norm(np.asarray(init_img) - np.asarray(tf_img))),
            }
        }
        with open(f"{save_dir}/index.yaml", "a+") as f:
            yaml.dump(info, f)
            f.write("\n")

        i += 1
        print(save_name)
        tf_img.save(save_name)

if __name__ == "__main__":
    main()
