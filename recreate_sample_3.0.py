import cv2
import numpy as np
import random
import itertools
import pickle
import argparse

##-------- by Dingrui Liu- ---------- -##
## Design ideas:
# 1, provide interactive operation of circle and triangle sampling
# 2, the triangle has two types: right triangle and orthogonal triangle, which are controlled by angle parameter.
# 3, triangle, circle size has upper and lower bounds, respectively by rad parameter to control the
# 4, function: q exit, s save, b back to the previous step, after saving will automatically generate samples
# Changes: keep the initial sampling area for each picture.
# Change: add canny edges inside the mask
# Alteration: fuse circles and triangles in one image

number = 400

parser = argparse.ArgumentParser()
parser.add_argument("--png_name", default="dummy", help="Input image name")
args = parser.parse_args()

n = args.png_name

path_pic = "./img/"
p_name = n + ".png"  # Update
save_sample_path = "../EEC_save_sample_13.0/"
path_sor = save_sample_path + n + "/nms/"
path_lab = save_sample_path + n + "/source/"
path_msk = "../Dataset/EEC/EEC_test_dataset_label/" + n + ".png"  # Update
path_m_sor = "./seg/source_train/"
path_m_lab = "./seg/sample/"
path_train = "./Medical_Datasets/Images/"
path_gt = "./Medical_Datasets/Labels/"
# Edge
path_m_eg = "./seg/label/"
path_egt = "./Medical_Datasets/edges/"


class Label:
    def __init__(self, left, right, angle, path):
        self.r_min = left
        self.r_max = right
        self.angle = angle
        self.start = 1
        self.id = 0
        self.bg = 0  # Sample points used to distinguish target from background
        self.trnum = 0
        self.r = left
        self.point1 = (0, 0)
        self.num_class = 0
        self.value_initial = {"col": (255, 255, 255), "val": 1}
        self.colors = [
            (255, 0, 0),
            (0, 128, 0),
            (128, 128, 0),
            (0, 0, 128),
            (128, 0, 128),
            (0, 128, 128),
            (128, 128, 128),
            (64, 0, 0),
            (192, 0, 0),
            (64, 128, 0),
            (192, 128, 0),
            (64, 0, 128),
            (192, 0, 128),
            (64, 128, 128),
            (192, 128, 128),
            (0, 64, 0),
            (128, 64, 0),
            (0, 192, 0),
            (128, 192, 0),
            (0, 64, 128),
            (128, 64, 12),
        ]
        self.img = cv2.imread(path)
        self.sp = self.img.shape
        self.img2 = self.img.copy()
        self.slic = slic_(path)
        self.randlist = list(itertools.product(range(self.sp[1]), range(self.sp[0])))
        self.cent = []
        self.ind = {}
        self.tind = {}  #
        self.cnd = {}
        self.tcnd = {}  #
        self.rnd = {}
        try:
            with open(save_sample_path + n + "/rnd.pkl", "rb") as f8:
                try:
                    self.rnd = pickle.load(f8)
                except EOFError:
                    return None
        except FileNotFoundError:
            return None
        self.weight = {}

    def trans(self, hot, p_l, p_s, id, sc, c, r, typ, rot=True):
        rows, cols = self.sp[0], self.sp[1]
        lable = np.zeros((rows, cols, 3), np.uint8)
        lable_e = np.zeros((rows, cols, 3), np.uint8)
        r_h, c_h = rows // 2, cols // 2
        s = int(sc)
        if not id:
            point = random.choice(self.randlist)
        else:
            point_lst = hot.copy()
            point_reverse = random.choice(point_lst)
            point = (point_reverse[1], point_reverse[0])
        my, mx = c_h - c[0], r_h - c[1]
        my_, mx_ = point[0] - c_h, point[1] - r_h
        M = np.float32([[1, 0, my], [0, 1, mx]])
        M_ = np.float32([[1, 0, my_], [0, 1, mx_]])
        rot = random.randint(0, 1)
        if typ:
            angle = 0
        else:
            angle = random.randint(1, 360) if rot else 0
        R = cv2.getRotationMatrix2D((cols // 2, rows // 2), angle, 1)
        if typ == 0:
            if 255 in p_l:
                dst_l = cv2.circle(lable, point, r - 1, (s - 1, s - 1, s - 1), -1)
                dst_e = cv2.circle(lable_e, point, r - 1, (255, 255, 255), 1)
            else:
                dst_l = cv2.circle(lable, point, r - 1, (128, 128, 128), -1)
        if typ == 1:
            dst_l = np.zeros((rows, cols, 3), np.uint8)
            dst_e = np.zeros((rows, cols, 3), np.uint8)
            if 255 in p_l:
                dst_l[p_l >= 254] = s - 1
                dst_e[p_l == 254] = 255
            else:
                dst_l[p_l >= 127] = 128
            dst_l = cv2.warpAffine(dst_l, M, (cols, rows), borderValue=(0, 0, 0))
            dst_e = cv2.warpAffine(dst_e, M, (cols, rows), borderValue=(0, 0, 0))
            dst_l = cv2.warpAffine(dst_l, M_, (cols, rows), borderValue=(0, 0, 0))
            dst_e = cv2.warpAffine(dst_e, M_, (cols, rows), borderValue=(0, 0, 0))
        dst_s = cv2.warpAffine(p_s, M, (cols, rows), borderValue=(0, 0, 0))
        dst_s = cv2.warpAffine(dst_s, R, (cols, rows))
        dst_s = cv2.warpAffine(dst_s, M_, (cols, rows), borderValue=(0, 0, 0))
        return point, dst_l, dst_s, dst_e

    def creat(self, cind, ind, msk, ns, nl, th=0, mode=0):
        num_poi = len(cind)
        idx = np.zeros((len(cind), len(ind[1]) + 1))
        ms = {}
        for h in range(num_poi):
            ms[h + 1] = {}
            for j in range(len(ind[h + 1])):
                hot_idx = np.nonzero(msk[h + 1][j, :, :] == 0)
                ms[h + 1][j] = np.transpose([hot_idx[0], hot_idx[1]])
        for i in range(ns, nl):
            j = 0
            dif = max(8, len(cind))
            p, r = [], []
            edge_c = np.zeros(self.img.shape, np.uint8)
            label_c = np.zeros(self.img.shape, np.uint8)
            source_c = np.zeros(self.img.shape, np.uint8)
            while j < dif:
                typ = random.randint(0, 1)
                type_ = "t" if typ else "c"
                nc = mode if mode else random.randint(1, num_poi)
                cind_ = cind[nc]
                ind_ = ind[nc]
                scale_list = np.arange(1, len(ind_) + 1, 1)
                scalec = np.random.choice(scale_list)
                kindc = random.randint(1, ind_[scalec][1])
                r_c = ind_[scalec][0]
                c_c = cind_[scalec][kindc - 1]
                lab_c = cv2.imread(
                    path_lab
                    + type_
                    + "-"
                    + str(nc)
                    + "-"
                    + str(scalec)
                    + "-"
                    + str(kindc)
                    + ".png"
                )
                pic_c = cv2.imread(
                    path_sor
                    + type_
                    + "-"
                    + str(nc)
                    + "-"
                    + str(scalec)
                    + "-"
                    + str(kindc)
                    + ".png"
                )

                scalec_ = self.rnd[nc] if scalec == 1 else scalec
                poi, dstc_l, dstc_s, dstc_e = self.trans(
                    ms[nc][scalec - 1],
                    lab_c,
                    pic_c,
                    idx[nc - 1, scalec - 1],
                    scalec_,
                    c_c,
                    r_c,
                    typ,
                )
                if local_(poi, r_c, p, r):
                    label_c = label_c + dstc_l
                    source_c = source_c + dstc_s
                    edge_c = edge_c + dstc_e
                    p.append(poi)
                    r.append(r_c)
                    j += 1
                    msk[nc][scalec - 1, :, :] = np.array(
                        msk[nc][scalec - 1, :, :] + dstc_l[:, :, 0] / 255, np.uint16
                    )
                    if th != 0:
                        idx[nc - 1, scalec - 1] = np.sum(
                            np.sum(msk[nc][scalec - 1, :, :] == 0)
                        )
            cv2.imwrite(path_m_lab + str(i + 1) + ".png", label_c)
            cv2.imwrite(path_m_sor + str(i + 1) + ".png", source_c)
            cv2.imwrite(path_m_eg + str(i + 1) + ".png", edge_c)
        return msk

    def get_result(self, cind, tind, ind, indt, mode=0):
        mask = {}
        scales = len(ind[1]) + 1
        for j in range(1, len(cind) + 1):  # initialisation
            mask[j] = np.zeros((scales, self.sp[0], self.sp[1]), np.uint16)
        if mode == 0:
            num = number
            ns, nl = 0, num * 3
            img_hot_0 = self.creat(cind, ind, mask, ns, nl)
            for i in range(1, len(cind) + 1):
                for j in range(len(ind[i])):
                    img_hot = img_hot_0[i][j, :, :]
                    if np.count_nonzero(img_hot) < self.sp[0] * self.sp[1] * 0.75:
                        img_hot[img_hot != 0] = 1
                    else:
                        img_hot_plus = np.sort(img_hot.reshape(-1))
                        key_value = img_hot_plus[int(self.sp[0] * self.sp[1] * 0.25)]
                        img_hot[img_hot > key_value] = 1
                        img_hot[img_hot < key_value + 1] = 0
                    mask[i][j, :, :] = img_hot.copy()

            nns, nnl = 3 * num, 4 * num
            img_hot_01 = self.creat(cind, ind, mask, nns, nnl, 10)
            hot = {}
            for k in range(1, len(cind) + 1):
                hot[k] = img_hot_01[k]

    def run(self, n):
        try:
            with open(save_sample_path + n + "/cent.pkl", "rb") as f1:
                try:
                    self.cent = pickle.load(f1)
                except EOFError:
                    return None
        except FileNotFoundError:
            return None

        try:
            with open(save_sample_path + n + "/ind.pkl", "rb") as f2:
                try:
                    self.ind = pickle.load(f2)
                except EOFError:
                    return None
        except FileNotFoundError:
            return None

        try:
            with open(save_sample_path + n + "/cnd.pkl", "rb") as f3:
                try:
                    self.cnd = pickle.load(f3)
                except EOFError:
                    return None
        except FileNotFoundError:
            return None

        try:
            with open(save_sample_path + n + "/tcnd.pkl", "rb") as f4:
                try:
                    self.tcnd = pickle.load(f4)
                except EOFError:
                    return None
        except FileNotFoundError:
            return None

        try:
            with open(save_sample_path + n + "/tind.pkl", "rb") as f5:
                try:
                    self.tind = pickle.load(f5)
                except EOFError:
                    return None
        except FileNotFoundError:
            return None

        try:
            with open(save_sample_path + n + "/sp.pkl", "rb") as f6:
                try:
                    self.sp = pickle.load(f6)
                except EOFError:
                    return None
        except FileNotFoundError:
            return None

        try:
            with open(save_sample_path + n + "/trnum.pkl", "rb") as f7:
                try:
                    self.trnum = pickle.load(f7)
                except EOFError:
                    return None
        except FileNotFoundError:
            return None

        print("Recrating " + n)
        print("wait...\n")
        self.get_result(self.cnd, self.tcnd, self.ind, self.tind, 0)
        # cv2.imwrite(path_pic+'sample_'+p_name, self.img2) # Record sampling images
        print("sampling finished,start building trainsets and segments!\n")
        comb(self.cent, path_pic + p_name, path_msk, path_m_lab, path_m_sor, path_train)

        level = (self.r_max - self.r_min) // 4 + 1
        creat_seg(
            self.cent,
            path_m_lab,
            path_m_eg,
            path_gt,
            path_msk,
            path_egt,
            self.sp,
            self.trnum,
            level,
            1,
        )  # 1 is dichotomous, 0 is multichotomous
        print("process finished！\n")


# external function


def Cal_area_circle(p_1, r_1, p_2, r_2):
    dis = (p_1[0] - p_2[0]) ** 2 + (p_1[1] - p_2[1]) ** 2
    return dis < (r_1 + r_2 + 1) ** 2


def local_(p_new, r_new, p, r):
    length = len(p)
    for i in range(length):
        if Cal_area_circle(p_new, r_new, p[i], r[i]):
            return False
    return True


def comb(clis, p_sc, p_ms, plab, ptrain, ftrain):
    # n_pic = number * len(clis) * 4
    n_pic = number * 4 - 1  # after adding the 'crescent moon'
    picture = cv2.imread(p_sc)
    for i in range(1, n_pic + 1):
        back = picture.copy()
        pic = cv2.imread(ptrain + str(i) + ".png")
        lab_ = cv2.imread(plab + str(i) + ".png")
        back[lab_ > 0] = pic[lab_ > 0]
        cv2.imwrite(ftrain + str(i) + ".jpg", back)
        # original figure
        copy = cv2.imread(path_pic + p_name)
        cv2.imwrite(ftrain + str(n_pic + 1) + ".jpg", copy)
    print("creat trainsets finished!\n")


def creat_seg(
    clis, path_s, path_e, path_d, path_ms, path_de, shape, trnum, level, type=1
):  # acquire the final gt
    n_pic = number * 4 - 1
    msk = cv2.imread(path_ms)
    mask = cv2.cvtColor(msk, cv2.COLOR_BGR2GRAY)
    for i in range(n_pic):
        pic = np.zeros((shape[0], shape[1]), np.uint8)
        edge = np.zeros((shape[0], shape[1]), np.uint8)
        sc = cv2.imread(path_s + str(i + 1) + ".png", 0)
        sc_e = cv2.imread(path_e + str(i + 1) + ".png", 0)
        pic[mask == 255] = 255
        edge[sc_e == 255] = 1
        edge[mask == 255] = 255
        for j in range(1, level + 1):
            pic[sc == j] = j
        pic[sc == 128] = 0
        cv2.imwrite(path_d + str(i + 1) + ".png", pic)
        cv2.imwrite(path_de + str(i + 1) + ".png", edge)
    # Circle the foci to train together
    pic_c = np.zeros((shape[0], shape[1]), np.uint8)
    edge_c = np.zeros((shape[0], shape[1]), np.uint8)
    pic_c[mask == 255] = 255
    edge_c[mask == 255] = 255
    for j in range(len(clis)):
        lab_c = cv2.imread(
            path_lab + "c" + "-" + str(j + 1) + "-" + "1" + "-" + "1" + ".png", 0
        )
        if j < trnum:
            pic_c[lab_c > 0] = level
        else:
            pic_c[lab_c == 128] = 0
    cv2.imwrite(path_d + str(n_pic + 1) + ".png", pic_c)
    cv2.imwrite(
        path_de + str(n_pic + 1) + ".png", edge_c
    )  # By default the outer edges are not recognised and the inner edges are not involved in the training, so at this point the edge gt is empty
    print("creat segs finished!\n")


def slic_(p_t):
    img = cv2.imread(p_t)
    # Initialise the slic item with an average superpixel size of 20 (default 10) and a smoothing factor of 20
    slic = cv2.ximgproc.createSuperpixelSLIC(img, region_size=48, ruler=20.0)
    slic.iterate(10)  # Number of iterations, the larger the better the result
    label_slic = np.array(slic.getLabels(), np.uint8)  # Get hyperpixel labels
    return label_slic


if __name__ == "__main__":
    lab = Label(12, 49, 0, path_pic + p_name)
    lab.run(n)
