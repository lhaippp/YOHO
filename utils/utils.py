import numpy as np
from PIL import Image

# Converts images to RGB images to prevent grey scale maps from reporting errors in prediction.
# The code only supports prediction of RGB images, all other types of images are converted to RGB.


def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[-2] == 3:
        return image
    else:
        image = image.convert("RGB")
        return image


#   Resize the input images


def resize_image(image, size):
    iw, ih = image.size
    w, h = size

    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new("RGB", size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))

    return new_image, nw, nh


#   Acquire the learning rate
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def preprocess_input(image):
    image /= 255.0
    return image
