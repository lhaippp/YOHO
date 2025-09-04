# Single-Image-Based Deep Learning for Segmentation of Early Esophageal Cancer Lesions

The test set for EEC-2022 can be found on: [EEC-2022](https://drive.google.com/file/d/1NeuGRLbicY2awAUW44uQl6BMu8S5feVy/view?usp=sharing) under the MIT License.

The full dataset for EEC-2022 could be found here: [Link](https://drive.google.com/file/d/1hBHj4dy_dESjmq496TbZpLYXKIpnrSww/view?usp=drive_link)

## 0. Early Esophageal Cancer (EEC) Segmentation Data Description
All gastroscopy image data were acquired using Olympus GIF-Q290 and GIF-Q260 electronic gastroscopes. 

The collected gastroscopy images include resolutions of: 1920 px × 1080 px, 768 px × 576 px, and 480 px × 360 px (in this dataset, cropped original images have been uniformly resized to 480×480; users may adjust dimensions according to their network capabilities). 

Non-EEC cases overlap with EEC cases, meaning non-EEC images were acquired from the non-EEC esophageal mucosa of EEC patients. Acquisition and use of the EEC (Early Esophageal Cancer) dataset were approved by the Ethics Committees of West China Hospital, Sichuan University and the University of Electronic Science and Technology of China. All cases obtained patient informed consent.

Images were annotated by experienced specialists. Ground truth (mask) uses pixel 0 for background and 1 for EEC, appearing entirely black. The non-EEC mask can be entirely 0 and may be created independently. Non-EEC images should be randomly assigned to the train, val, or test sets.

Data Details:
The train, val, and test datasets were randomly partitioned under the constraint that images from the same patient appear only in the same dataset. Do not perform further random splits on the data. If redistributing datasets is necessary, trace images from the same patient using EEC image filenames. For example, ‘2ESTD1jhx201492000OR.png’ and ‘2ESTD1jhx201492004OR.png’ are gastroscopy images from the same patient.

Image naming convention: 0.png-1299.png denote non-EEC images; all others are EEC images.


## 1. Preface
- Paper is available at [https://arxiv.org/abs/2306.05912](https://arxiv.org/abs/2306.05912).

## 2. Proposed Baseline

### 2.1. ROI
To generate this ROI plot,'labelme' is used, version 3.16.7.

Examples of ROI plots are shown in Figure 1. The white area (grey value of 255) is ROI, which is the range where the sample is located. Most of the time the sampled area is focal tissue (the first row of the examples), but for some images, where the extent of the lesion is more than 50% to 60% of the map, the normal tissue area is segmented, i.e., the ROI was chosen to be within the normal tissue area (the second row of the examples). One of the more obvious features of such images is that the four corners are white (grey value of 255). For such images, since the segmentation is the normal organisation, it is sufficient to take the complement of the final segmentation result.

<p align="center">
    <img src="illustrative_figure/ROI_example.png"/ width="50%" height="50%"> <br />
    <em> 
    Figure 1. Examples of ROI plots.
    </em>
</p>

### 2.2. Sampling

All the images you want to process can be put into `./img/`, which can facilitate subsequent operations. Record sampling information with `./interaction7_record_sample_3.0.py`. The sampling information saved are 2 files `nms`, `source` and 8 PKL files: `cent.pkl`, `cnd.pkl`, `ind.pkl`, `rnd.pkl`, `sp.pkl`, `tcnd.pkl`, `tind.pkl`, `trnum.pkl`.

### 2.3. Generating training sets, training and prediction

After collecting the information of all the pictures and setting the parameter information of different scripts, the operation of 'Generate dataset - Training - Prediction' can be realised in batch by running `./Start.py` script.

`./recreate_sample_3.0.py` is to generate the training set based on the information recorded from the previous sampling.

`./voc_annotation_medical.py` is to generate txt for the training set.

`./train_medical.py` is for training. The weights files will be saved in `./logs/`. If you want to change where the weights file is stored, check out `./utils/utils_fit.py/`.

The path to save the final segmented mask can be modified in `./unet.py`. After running `./predict.py`, intuitive segmented image will be saved in `./img_out/`.



## 3. Citation
```
@misc{li2023singleimagebased,
      title={Single-Image-Based Deep Learning for Segmentation of Early Esophageal Cancer Lesions}, 
      author={Haipeng Li and Dingrui Liu and Yu Zeng and Shuaicheng Liu and Tao Gan and Nini Rao and Jinlin Yang and Bing Zeng},
      year={2023},
      eprint={2306.05912},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
```
