import os
import random

trainval_percent = 1
train_percent = 1

# Points to the folder where the dataset is located, by default it points to 'Medical_Datasets' in the root directory
VOCdevkit_path = "Medical_Datasets"

if __name__ == "__main__":
    random.seed(0)
    print("Generate txt in ImageSets.")
    segfilepath = os.path.join(VOCdevkit_path, "Labels")
    saveBasePath = os.path.join(VOCdevkit_path, "ImageSets/Segmentation")

    temp_seg = os.listdir(segfilepath)
    total_seg = []
    for seg in temp_seg:
        if seg.endswith(".png"):
            total_seg.append(seg)

    num = len(total_seg)
    list = range(num)
    tv = int(num * trainval_percent)
    tr = int(tv * train_percent)
    trainval = random.sample(list, tv)
    train = random.sample(trainval, tr)

    print("train and val size", tv)
    print("traib size", tr)
    ftrainval = open(os.path.join(saveBasePath, "trainval.txt"), "w")
    ftest = open(os.path.join(saveBasePath, "test.txt"), "w")
    ftrain = open(os.path.join(saveBasePath, "train.txt"), "w")
    fval = open(os.path.join(saveBasePath, "val.txt"), "w")

    for i in list:
        name = total_seg[i][:-4] + "\n"
        if i in trainval:
            ftrainval.write(name)
            if i in train:
                ftrain.write(name)
            else:
                fval.write(name)
        else:
            ftest.write(name)

    ftrainval.close()
    ftrain.close()
    fval.close()
    ftest.close()
    print("Generate txt in ImageSets done.")
