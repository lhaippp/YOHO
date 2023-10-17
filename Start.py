import os

if __name__ == "__main__":
    img_path = "./img/"
    imgList = os.listdir(img_path)
    for imgs in imgList:
        png_name = imgs.split(".")[0]

        cmd = "python recreate_sample_3.0.py --png_name {}".format(png_name)
        os.system(cmd)

        cmd = "python voc_annotation_medical.py"
        os.system(cmd)

        cmd = "python train_medical.py --png_name {}".format(png_name)
        os.system(cmd)

        cmd_temp = "python unet.py --png_name {}".format(png_name)

        cmd = "python predict.py --png_name {}".format(png_name)
        os.system(cmd)
