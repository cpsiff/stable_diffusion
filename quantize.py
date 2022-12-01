from PIL import Image
import os

for f in os.listdir("cropped_images"):
    img = Image.open(f"cropped_images/{f}")
    img = img.quantize(colors=16).convert("RGB")
    img.save(f"quantized/{f}")