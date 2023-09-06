import argparse
from PIL import Image

from unet import Unet

parser = argparse.ArgumentParser()
parser.add_argument("--png_name", default="dummy", help="Input image name")
args = parser.parse_args()

if __name__ == "__main__":
    unet = Unet()

    mode = "dir_predict"

    dir_origin_path = "img/"
    dir_save_path = "img_out/"

    if mode == "dir_predict":
        import os
        from tqdm import tqdm

        png_name = args.png_name
        img_name = args.png_name + ".png"
        image_path = os.path.join(dir_origin_path, img_name)
        image = Image.open(image_path)
        r_image = unet.detect_image(image, img_name, png_name)
        if not os.path.exists(dir_save_path):
            os.makedirs(dir_save_path)
        r_image.save(os.path.join(dir_save_path, img_name))

        # img_names = os.listdir(dir_origin_path)
        # for img_name in tqdm(img_names):
        #     if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
        #         image_path  = os.path.join(dir_origin_path, img_name)
        #         image       = Image.open(image_path)
        #         r_image     = unet.detect_image(image)
        #         if not os.path.exists(dir_save_path):
        #             os.makedirs(dir_save_path)
        #         r_image.save(os.path.join(dir_save_path, img_name))

    else:
        raise AssertionError("Please specify the correct mode: 'dir_predict'.")
