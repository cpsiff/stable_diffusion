import io
from PIL import Image

# takes a PIL image as input
def convert_to_jpg(img, qual):

    with io.BytesIO() as output:
        img.save(output, format="jpeg", quality=qual, subsampling=0)
        contents = output.getvalue()

    converted_img = Image.open(io.BytesIO(contents))
    return converted_img