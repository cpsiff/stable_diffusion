from PIL import Image
import os
import io

def convert_to_jpg(img, qual):
    with io.BytesIO() as output:
        img.save(output, format="jpeg", quality=qual, subsampling=0)
        contents = output.getvalue()

    converted_img = Image.open(io.BytesIO(contents))
    return converted_img

for f in os.listdir("cropped_images"):
    img = Image.open(f"cropped_images/{f}")
    img = convert_to_jpg(img, qual=10).convert("RGB")
    img.save(f"JPEGified/{f}")