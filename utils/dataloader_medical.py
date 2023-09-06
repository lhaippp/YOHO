import os

import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset

from utils.utils import cvtColor, preprocess_input


class UnetDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, train, dataset_path):
        super(UnetDataset, self).__init__()
        self.annotation_lines = annotation_lines
        self.length = len(annotation_lines)
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.train = train
        self.dataset_path = dataset_path

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        annotation_line = self.annotation_lines[index]
        name = annotation_line.split()[0]

        #   Read images from the files
        jpg = Image.open(
            os.path.join(os.path.join(self.dataset_path, "Images"), name + ".jpg")
        )
        png = Image.open(
            os.path.join(os.path.join(self.dataset_path, "Labels"), name + ".png")
        )
        edge = Image.open(
            os.path.join(os.path.join(self.dataset_path, "edges"), name + ".png")
        )

        #   Data Enhancement
        jpg, png, edge = self.get_random_data(
            jpg, png, edge, self.input_shape, random=False
        )
        jpg = np.transpose(preprocess_input(np.array(jpg, np.float64)), [2, 0, 1])
        png_t = np.array(png)
        edge = np.array(edge)

        png_t[np.array(png) < 255] = 1
        png_t[np.array(png) == 0] = 0
        png_t[np.array(png) == 255] = 2

        png_w = np.zeros(np.shape(png), np.float64)

        seg_labels = np.eye(self.num_classes + 1)[png_t.reshape([-1])]
        seg_labels = seg_labels.reshape(
            (int(self.input_shape[0]), int(self.input_shape[1]), self.num_classes + 1)
        )

        # Identical processing of edges to facilitate dice_loss calculations
        edge[edge > 128] = 2

        return jpg, png_t, png_w, seg_labels, edge

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(
        self,
        image,
        label,
        edge,
        input_shape,
        jitter=0.3,
        hue=0.1,
        sat=1.5,
        val=1.5,
        random=True,
    ):
        image = cvtColor(image)
        label = Image.fromarray(np.array(label))
        edge = Image.fromarray(np.array(edge))
        h, w = input_shape

        if not random:
            iw, ih = image.size
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)

            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new("RGB", [w, h], (128, 128, 128))
            new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))

            label = label.resize((nw, nh), Image.NEAREST)
            new_label = Image.new("L", [w, h], (0))
            new_label.paste(label, ((w - nw) // 2, (h - nh) // 2))

            edge = edge.resize((nw, nh), Image.NEAREST)
            new_edge = Image.new("L", [w, h], (0))
            new_edge.paste(edge, ((w - nw) // 2, (h - nh) // 2))

        return new_image, new_label, new_edge


def unet_dataset_collate(batch):
    # Add edge information
    images = []
    pngs = []
    pngs_w = []
    seg_labels = []
    edges = []

    for img, png, png_w, labels, edge in batch:
        images.append(img)
        pngs.append(png)
        pngs_w.append(png_w)
        seg_labels.append(labels)
        edges.append(edge)
    images = np.array(images)
    pngs = np.array(pngs)
    pngs_w = np.array(pngs_w)
    seg_labels = np.array(seg_labels)
    edges = np.array(edges)
    return images, pngs, pngs_w, seg_labels, edges


def do_center_pad(image, mask, weight, edge, pad_left, pad_right):
    image = np.pad(
        image, ((0, 0), (pad_left, pad_right), (pad_left, pad_right)), "edge"
    )
    mask = np.pad(mask, (pad_left, pad_right), "edge")
    edge = np.pad(edge, (pad_left, pad_right))
    weight = np.pad(weight, (pad_left, pad_right))
    return image, mask, weight, edge
