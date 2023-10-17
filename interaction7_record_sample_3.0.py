from typing import Dict, Union, List, Tuple, Any

import cv2
import numpy as np
import random
import itertools
import pickle
import math
import os

##-------- by Dingrui Liu- ---------- -##
## Design ideas:
# 1, provide interactive operation of circle and triangle sampling
# 2, the triangle has two types: right triangle and orthogonal triangle, which are controlled by angle parameter.
# 3, triangle, circle size has upper and lower bounds, respectively by rad parameter to control the
# 4, function: q exit, s save, b back to the previous step, after saving will automatically generate samples
# Changes: keep the initial sampling area for each picture.
# Change: add canny edges inside the mask
# Alteration: fuse circles and triangles in one image


n = str(30)

path_pic = "../EEC/test/"
p_name = n + ".png"

# Saved sampling information  
save_sample_path = (
    "../EEC/EEC_save_sampling_information/"
)
path_sor = save_sample_path + n + "/nms/"
path_lab = save_sample_path + n + "/source/"

# Intermediate generation of information
path_m_sor = "../Seg/source_train/"
path_m_lab = "../Seg/sample/"
path_m_eg = "../Seg/label/"



number = 2


class Label:
    def __init__(self, left, right, angle, path):
        self.r_min = left
        self.r_max = right
        self.angle = angle
        self.start = 1
        self.id = 0
        self.bg = 0  # Sample points used to differentiate between target and background
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
        self.weight = {}

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:  # left click
            if self.num_class < 20:
                if self.start:
                    self.point1 = (x, y)
                    self.r = self.r_min
                    cv2.circle(
                        self.img2, self.point1, self.r, self.colors[self.num_class], -1
                    )

                elif self.r < self.r_max:
                    self.r = self.r + 1
                    cv2.circle(
                        self.img2, self.point1, self.r, self.colors[self.num_class], -1
                    )

        elif event == cv2.EVENT_LBUTTONUP:  # Left click to release
            self.start = 0

        elif event == cv2.EVENT_RBUTTONDOWN:  # right click
            if self.r > self.r_min:
                pic_now = cv2.circle(
                    np.zeros(self.img2.shape, np.uint8),
                    self.point1,
                    self.r,
                    (1, 1, 1),
                    -1,
                )
                self.img2 = self.img * pic_now + self.img2 * (1 - pic_now)
                self.r = self.r - 1
                cv2.circle(
                    self.img2, self.point1, self.r, self.colors[self.num_class], -1
                )

    def sample(self, poi, rad, id, iter_step=4, mode=0):
        r_min = self.r_min if mode == 0 else self.r_min * 2 - 1
        r_max = self.r_max
        scale = 0
        color_ = (255, 255, 255) if mode == 0 else (128, 128, 128)
        dic = {}
        tdic = {}
        c_ind = {}
        t_ind = {}
        lst = list(range(r_max, r_min - 1, -iter_step))
        scale_r = max(math.ceil((r_max - rad) / iter_step), 2)
        if rad in lst:
            lst.remove(rad)
        lst.insert(0, rad)
        weight = np.zeros((len(lst)))
        weight_normal = np.zeros((len(lst)))
        for i, R in enumerate(lst):
            weight[i] = R * R
        for i in range(len(lst)):
            msum = 0
            for j in range(len(lst)):
                msum += weight[i] / weight[j]
            weight_normal[i] = 1 / msum
        for r in lst:
            scale += 1
            area = np.zeros(self.img.shape, dtype=np.uint8)
            sp = pow(math.ceil(rad / r), 2)
            dic[scale] = (r, sp)
            tdic[scale] = (r, sp)
            # circles
            if r > rad:
                p_new = [(self.sp[1] // 2, self.sp[0] // 2)]
            else:
                cv2.circle(area, poi, rad - r, (255, 255, 255), -1)
                idx = np.nonzero(area == 255)
                idx_list = list(zip(idx[1], idx[0]))
                p_new = random.sample(idx_list, sp)

            # triangle

            c_ind[scale] = p_new
            t_ind[scale] = p_new
            for i, c in enumerate(p_new):
                scope = self.img.copy()
                mask = np.zeros(self.img.shape, dtype=np.uint8)
                cv2.circle(mask, c, r, color_, -1)
                cv2.imwrite(
                    path_lab
                    + "c"
                    + "-"
                    + str(id)
                    + "-"
                    + str(scale)
                    + "-"
                    + str(i + 1)
                    + ".png",
                    mask,
                )  # The label of circle
                if r > rad:
                    scope = get_sc(scope, mask, r, rad, poi)
                else:
                    scope[mask == 0] = 0
                cv2.imwrite(
                    path_sor
                    + "c"
                    + "-"
                    + str(id)
                    + "-"
                    + str(scale)
                    + "-"
                    + str(i + 1)
                    + ".png",
                    scope,
                )  # the source of circle

            for i, c in enumerate(p_new):
                r_t = r // 2
                mask_t, mask_e = self.tri(c, r_t, color_)
                mask_t = mask_t - mask_e
                cv2.imwrite(
                    path_lab
                    + "t"
                    + "-"
                    + str(id)
                    + "-"
                    + str(scale)
                    + "-"
                    + str(i + 1)
                    + ".png",
                    mask_t,
                )  # The label of triangle
                scope_t = self.img.copy()
                if r > rad:
                    scope_t = get_sc(scope_t, mask_t, r, rad, poi)
                else:
                    scope_t[mask_t == 0] = 0
                cv2.imwrite(
                    path_sor
                    + "t"
                    + "-"
                    + str(id)
                    + "-"
                    + str(scale)
                    + "-"
                    + str(i + 1)
                    + ".png",
                    scope_t,
                )  # The source of triangle

        return dic, tdic, c_ind, t_ind, scale_r, weight_normal

    def tri(self, center, r_t, col):
        pic_ = np.zeros(self.sp, np.uint8)
        cv2.circle(pic_, center, r_t, (255, 255, 255), -1)
        pic_edge = np.zeros(self.sp, np.uint8)
        gray = cv2.cvtColor(pic_, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, 1, 2)
        c = np.array(contours[0], np.float32)
        area, trg = cv2.minEnclosingTriangle(c)
        trg = np.array(trg, np.int32)
        cv2.fillPoly(pic_, [trg], col)
        cv2.polylines(pic_edge, [trg], isClosed=True, color=(1, 1, 1), thickness=1)
        return pic_, pic_edge

    def rec(self, center, r_t, col):
        pic_ = np.zeros(self.sp, np.uint8)
        cv2.circle(pic_, center, r_t, (255, 255, 255), -1)
        pic_edge = np.zeros(self.sp, np.uint8)
        gray = cv2.cvtColor(pic_, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, 1, 2)
        c = np.array(contours[0], np.float32)
        retval = cv2.minAreaRect(c)
        points = cv2.boxPoints(retval)
        points = np.array(points, np.int32)
        cv2.fillPoly(pic_, [points], col)
        cv2.polylines(pic_edge, [points], isClosed=True, color=(1, 1, 1), thickness=1)
        return pic_, pic_edge

    def pentagon(self, center, r_t, col):
        pic_ = np.zeros(self.sp, np.uint8)
        pic_edge = np.zeros(self.sp, np.uint8)
        pen_point = []
        point_1 = [center[0] + r_t, center[1]]
        pen_point.append(point_1)
        point_2 = [center[0] + r_t / 2, center[1] - r_t / 2 * math.sqrt(3)]
        pen_point.append(point_2)
        point_3 = [center[0] - r_t / 2, center[1] - r_t / 2 * math.sqrt(3)]
        pen_point.append(point_3)
        point_4 = [center[0] - r_t, center[1]]
        pen_point.append(point_4)
        point_5 = [center[0], center[1] + r_t]
        pen_point.append(point_5)
        pen_point = np.array(pen_point, np.int32)
        cv2.fillPoly(pic_, [pen_point], col)
        cv2.polylines(
            pic_edge, [pen_point], isClosed=True, color=(1, 1, 1), thickness=1
        )
        return pic_, pic_edge

    def convex(self, point, rad, col):
        mask_old = np.zeros(self.img.shape, dtype=np.uint8)
        mask_new = np.zeros(self.img.shape, dtype=np.uint8)
        cv2.circle(mask_old, point, rad, col, -1)
        move_list = [[0, rad], [rad, 0], [0, -rad], [-rad, 0]]
        move = random.choice(move_list)
        p = list(point)
        p[0], p[1] = p[0] + move[0], p[1] + move[1]
        p = tuple(p)
        cv2.circle(mask_new, p, rad, col, -1)
        piccc = cv2.bitwise_and(mask_old, (col[0] - mask_new))
        return piccc

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

    def creat(self, cind, ind, msk, typ, ns, nl, th=0, mode=0):
        if typ == 0:
            type_ = "c"
        elif typ == 1:
            type_ = "t"
        else:
            type_ = "v"
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
                #
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

    def get_result(self, cind, tind, ind, indt, mode):
        mask = {}
        mask_t = {}
        scales = len(ind[1]) + 1
        for j in range(1, len(cind) + 1):
            mask[j] = np.zeros((scales, self.sp[0], self.sp[1]), np.uint16)
            mask_t[j] = np.zeros((scales, self.sp[0], self.sp[1]), np.uint16)
        if mode == 0:
            num = number
            ns, nl = 0, num
            ns_, nl_ = num, 3 * num
            img_hot_0 = self.creat(cind, ind, mask, 0, ns, nl)
            img_hot_1 = self.creat(tind, indt, mask_t, 1, ns_, nl_)
            for i in range(1, len(cind) + 1):
                for j in range(len(ind[i])):
                    img_hot = np.zeros((self.sp[0], self.sp[1]), np.uint16)
                    img_hot = img_hot_0[i][j, :, :] + img_hot_1[i][j, :, :]
                    if np.count_nonzero(img_hot) < self.sp[0] * self.sp[1] * 0.75:
                        img_hot[img_hot != 0] = 1
                    else:
                        img_hot_plus = np.sort(img_hot.reshape(-1))
                        key_value = img_hot_plus[int(self.sp[0] * self.sp[1] * 0.25)]
                        img_hot[img_hot > key_value] = 1
                        img_hot[img_hot < key_value + 1] = 0
                    mask[i][j, :, :] = img_hot.copy()

            nns, nnl = 3 * num, 3 * num + num // 2
            nns_, nnl_ = 3 * num + num // 2, 4 * num - 1
            img_hot_01 = self.creat(cind, ind, mask, 0, nns, nnl, 10)
            img_hot_11 = self.creat(tind, indt, mask, 1, nns_, nnl_, 10)
            hot = {}
            for k in range(1, len(cind) + 1):
                # hot[k] = img_hot_01[k]
                # hot[k] = img_hot_11[k]
                hot[k] = img_hot_01[k] + img_hot_11[k]

    def update_(self, img, sl, ed, p, r, c):
        bg = np.zeros((self.sp[0], self.sp[1], 3), np.uint8)
        cv2.circle(bg, p, r, (255, 255, 255), -1)
        cg = sl[bg[:, :, 0] == 255]
        lst = np.unique(cg)
        for i in lst:
            if sum(cg == i) > 10:
                ed[sl == i] = 255
            # ed = cv2.medianBlur(ed, 5)
        _, bina = cv2.threshold(ed, 1, 255, cv2.THRESH_BINARY)
        ##去掉旧的轮廓
        blank = np.zeros((self.sp[0], self.sp[1], 3), np.uint8)
        cv2.drawContours(blank, c, -1, (255, 255, 255), 1)
        img[blank == 255] = self.img[blank == 255]
        contours, _ = cv2.findContours(bina, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, (124, 255, 255), 1)
        return contours

    def del_files(self, dir_path):
        for root, dirs, files in os.walk(dir_path, topdown=False):
            print(root)
            print(dirs)
            print(files)
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))

    def run(self):
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", self.on_mouse)
        # 定义slic轮廓
        edge_show = np.zeros((self.sp[0], self.sp[1]), np.uint8)
        ctr = []
        while True:
            cv2.imshow("image", self.img2)
            k = cv2.waitKey(1)

            if k == ord("q"):  # 退出程序
                if self.bg == 0:
                    print("病灶采集完成，请继续采集背景区域，按q退出\n")
                    self.trnum = self.num_class
                    self.bg = 1
                else:
                    break
            if k == ord("s"):  # 保存结果
                self.num_class += 1
                self.id += 1
                self.start = 1
                self.cent.append(self.point1)  #
                index, tindex, c_index, tc_index, s_r, wgt = self.sample(
                    self.point1, self.r, self.id, 4, self.bg
                )
                self.ind[self.num_class] = index
                self.tind[self.num_class] = tindex  # 三角形
                self.cnd[self.num_class] = c_index
                self.tcnd[self.num_class] = tc_index  # 三角形
                self.rnd[self.num_class] = s_r
                self.weight[self.num_class] = wgt
                if self.bg == 0:
                    ctr = self.update_(
                        self.img2, self.slic, edge_show, self.point1, self.r, ctr
                    )
                print("successfully saved one sample!\n")

            if k == ord("b"):  # 撤销上一步操作
                blank = np.zeros((self.sp[0], self.sp[1], 3), np.uint8)
                pic_now = cv2.circle(blank, self.point1, self.r, (1, 1, 1), -1)
                self.img2 = self.img * pic_now + self.img2 * (1 - pic_now)
                self.start = 1

        print("是否需要生成数据集？（y or n）\n")
        if input("input:") == "y":
            with open(save_sample_path + n + "/cent.pkl", "wb") as f1:  # cent 为采样点圆心位置
                pickle.dump(self.cent, f1)
            with open(save_sample_path + n + "/ind.pkl", "wb") as f2:
                pickle.dump(self.ind, f2)
            with open(save_sample_path + n + "/cnd.pkl", "wb") as f3:
                pickle.dump(self.cnd, f3)
            with open(save_sample_path + n + "/tcnd.pkl", "wb") as f4:
                pickle.dump(self.tcnd, f4)
            with open(save_sample_path + n + "/tind.pkl", "wb") as f5:
                pickle.dump(self.tind, f5)
            with open(save_sample_path + n + "/sp.pkl", "wb") as f6:
                pickle.dump(self.sp, f6)
            with open(save_sample_path + n + "/trnum.pkl", "wb") as f7:
                pickle.dump(self.trnum, f7)
            with open(save_sample_path + n + "/rnd.pkl", "wb") as f8:  # rnd 为最大的半径
                pickle.dump(self.rnd, f8)

            print("wait...\n")
            self.get_result(self.cnd, self.tcnd, self.ind, self.tind, 0)
            if not (os.path.exists(save_sample_path + "/sample_images")):
                os.mkdir(save_sample_path + "/sample_images")
            cv2.imwrite(
                save_sample_path + "/sample_images/" + p_name, self.img2
            )  # 记录采样图片
            print("sampling finished,start building trainsets and segments!\n")
            # comb(self.cent, path_pic+p_name, path_msk, path_m_lab, path_m_sor, path_train)
            #
            # level = (self.r_max-self.r_min)//4+1
            # creat_seg(self.cent, path_m_lab, path_m_eg, path_gt, path_msk, path_egt, self.sp, self.trnum, level, 1)  # 1是二分类，0是多分类
            # print('process finished！')

        else:
            self.del_files(path_sor)
            self.del_files(path_lab)


# 外部功能函数


def Cal_area_circle(p_1, r_1, p_2, r_2):
    dis = (p_1[0] - p_2[0]) ** 2 + (p_1[1] - p_2[1]) ** 2
    return dis < (r_1 + r_2 + 1) ** 2


def local_(p_new, r_new, p, r):
    length = len(p)
    for i in range(length):
        if Cal_area_circle(p_new, r_new, p[i], r[i]):
            return False
    return True


def inside(pic, mas, th, mode=0):
    if mode == 0:
        return False
    else:
        mask_gray = cv2.cvtColor(1 - mas, cv2.COLOR_BGR2GRAY)
        pic_gray = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
        res = cv2.bitwise_and(pic_gray, mask_gray)
        mix = np.count_nonzero(res)
    return mix <= th


def comb(clis, p_sc, p_ms, plab, ptrain, ftrain):
    # n_pic = number * len(clis) * 4
    n_pic = number * 4 - 1  # 添加弯月后
    picture = cv2.imread(p_sc)
    for i in range(1, n_pic + 1):
        back = picture.copy()
        pic = cv2.imread(ptrain + str(i) + ".png")
        lab_ = cv2.imread(plab + str(i) + ".png")
        back[lab_ > 0] = pic[lab_ > 0]
        cv2.imwrite(ftrain + str(i) + ".jpg", back)
        # 原图
        copy = cv2.imread(path_pic + p_name)
        cv2.imwrite(ftrain + str(n_pic + 1) + ".jpg", copy)
    print("creat trainsets finished!\n")


def creat_seg(
    clis, path_s, path_e, path_d, path_ms, path_de, shape, trnum, level, type=1
):  # 获取最终gt
    # n_pic = number *len(clis) * 4
    n_pic = number * 4 - 1
    msk = cv2.imread(path_ms)
    mask = cv2.cvtColor(msk, cv2.COLOR_BGR2GRAY)
    for i in range(n_pic):
        pic = np.zeros((shape[0], shape[1]), np.uint8)
        edge = np.zeros((shape[0], shape[1]), np.uint8)
        # col = 1
        sc = cv2.imread(path_s + str(i + 1) + ".png", 0)
        sc_e = cv2.imread(path_e + str(i + 1) + ".png", 0)
        pic[mask == 255] = 255
        edge[sc_e == 255] = 1
        edge[mask == 255] = 255
        # edge[sc_e == 255] = 1
        for j in range(1, level + 1):
            pic[sc == j] = j
        pic[sc == 128] = 0
        cv2.imwrite(path_d + str(i + 1) + ".png", pic)
        cv2.imwrite(path_de + str(i + 1) + ".png", edge)
    # 圈定的病灶一起训练
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
    )  # 默认不识别外部边缘，而内部边缘不参与训练，所以此时边缘gt是空的
    print("creat segs finished!\n")


def get_sc(pic, mask, r1, r2, c):
    pic_ = pic.copy()
    h, w = mask.shape[0], mask.shape[1]
    # print(h,w)
    rate = r1 / r2
    nh, nw = int(h * rate), int(w * rate)
    # print(nh,nw)
    my, mx = w // 2 - c[0], h // 2 - c[1]
    M = np.float32([[1, 0, my], [0, 1, mx]])
    pic_ = cv2.warpAffine(pic_, M, (w, h), borderValue=(0, 0, 0))
    pic_ = cv2.resize(pic_, (nw, nh), interpolation=cv2.INTER_CUBIC)
    h_new_l, h_new_r, w_new_l, w_new_r = (
        nh // 2 - h // 2,
        nh // 2 + h // 2,
        nw // 2 - w // 2,
        nw // 2 + w // 2,
    )
    if h % 2 != 0 or w % 2 != 0:
        h_new_r = h_new_l + h
        w_new_r = w_new_l + w
    # print(h_new_l,h_new_r,w_new_l,w_new_r)
    pico = pic_[h_new_l:h_new_r, w_new_l:w_new_r, :]
    pico[mask == 0] = 0
    return pico


def get_true(src):
    pic = cv2.imread(src, 0)
    gt = cv2.imread("C:/Users/Admin/Desktop/seg/true_20000.png")  # 临时添加
    mask = np.zeros((gt.shape[0], gt.shape[1]), np.uint8)
    mask[pic < 40] = 255
    mask = cv2.medianBlur(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), 3)
    mask = cv2.bitwise_and(mask, gt)
    return mask


def slic_(p_t):
    img = cv2.imread(p_t)
    # 初始化slic项，超像素平均尺寸20（默认为10），平滑因子20
    slic = cv2.ximgproc.createSuperpixelSLIC(img, region_size=48, ruler=20.0)
    slic.iterate(10)  # 迭代次数，越大效果越好
    label_slic = np.array(slic.getLabels(), np.uint8)  # 获取超像素标签
    return label_slic


if __name__ == "__main__":
    # 建议取值为(max(h,w)/256*4+1,max(h,w)/256*32+1)
    if not (os.path.exists(save_sample_path + n)):
        os.mkdir(save_sample_path + n)
    if not (os.path.exists(save_sample_path + n + "/nms")):
        os.mkdir(save_sample_path + n + "/nms")
    if not (os.path.exists(save_sample_path + n + "/source")):
        os.mkdir(save_sample_path + n + "/source")
    img = cv2.imread(path_pic + p_name)
    r_min = int(max(img.shape[0], img.shape[1]) / 256 * 8)
    if r_min % 2 != 0:
        r_min = r_min + 1
    r_max = 4 * r_min
    print("img_shape = {}, r_min = {}, r_max = {}".format(img.shape[0:2], r_min, r_max))
    lab = Label(r_min, r_max, 0, path_pic + p_name)
    lab.run()
