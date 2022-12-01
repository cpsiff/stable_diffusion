"""
Center crop all the source images down to 512x512
"""

from PIL import Image
from PIL import ImageOps
import os

SOURCE_DIR = "source_images"
OUT_DIR = "cropped_images"
WIDTH = 512
HEIGHT = 512

for f in os.listdir(SOURCE_DIR):
    im = Image.open(os.path.join(SOURCE_DIR, f))

    # resize image
    w, h = im.size
    aspect = w/h
    if w > h:
        new_h = HEIGHT
        new_w = aspect*new_h
    else:
        new_w = WIDTH
        new_h = (1/aspect)*new_w

    im = im.resize((int(new_w), int(new_h)), Image.Resampling.BILINEAR)

    im = ImageOps.exif_transpose(im)

    w, h = im.size
    left = (w - WIDTH)/2
    top = (h - HEIGHT)/2
    right = (w + WIDTH)/2
    bottom = (h + HEIGHT)/2

    # Crop the center of the image
    im = im.crop((left, top, right, bottom))

    im.save(os.path.join(OUT_DIR, f))