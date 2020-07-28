# This is built on the tutorial K-Means++ code from the Data Science Lab
# https://datasciencelab.wordpress.com/2014/01/15/improved-seeding-for-clustering-with-k-means/
import os
import numpy as np
import copy
import random
from mmdet.ops.nms.oks_nms_py import oks_iou
import mmcv
import torch


KMEANS_DIST_THRE = 0.01
KEEP_POSE_NOT_LESS_THAN = 17
LEFT_SHOULDER_ID = 5
RIGHT_SHOULDER_ID = 6


def mean_default(pts):
    return np.mean(pts, axis=0)


def mean_kpt(_pts):
    # TODO(xiao): other choices
    eps = 1e-5
    pts = copy.deepcopy(_pts)
    x = []
    y = []
    v = []
    for p in pts:
        x.append(p[:, 0])
        y.append(p[:, 1])
        p[p[:, 2] > 0, 2] = 1
        v.append(p[:, 2])
    v_sum = np.sum(v, axis=0) + eps
    x_mean = np.sum(x, axis=0) / v_sum
    y_mean = np.sum(y, axis=0) / v_sum
    v_mean = v_sum
    v_mean[v_mean >= 0.5] = 1
    v_mean[v_mean < 0.5] = 0
    return np.concatenate((x_mean.reshape((-1, 1)), y_mean.reshape((-1, 1)), v_mean.reshape((-1, 1))), axis=1)


def dist_default(x, c):
    return np.linalg.norm(x - c)


def dist_kpt(x, c):
    # TODO(xiao): other choices
    dist = oks_iou(x.reshape((-1)), c.reshape((1, -1)), cal_area(x), np.asarray(cal_area(c)).reshape((1, -1)),
                   in_vis_thre=0.5)
    dist = 1.0 / dist
    dist = np.log(dist)
    dist = np.sqrt(dist)
    return dist


def cal_area(x):
    if x.shape[-1] == 2:
        valid = x[:, 0] * 0 == 0
    elif x.shape[-1] == 3:
        valid = x[:, 2] > 0
    else:
        assert 0
    v = x[valid]
    if len(v) < 2:
        return x[0][0] * 0
    # TODO(xiao): plus one?
    w = max(v[:, 0]) - min(v[:, 0])
    h = max(v[:, 1]) - min(v[:, 1])
    return w * h


def cal_area_torch(v):
    # TODO(xiao): plus one?
    w = torch.max(v[:, :, 0], -1)[0] - torch.min(v[:, :, 0], -1)[0]
    h = torch.max(v[:, :, 1], -1)[0] - torch.min(v[:, :, 1], -1)[0]
    return w * h


def cal_area_2(x):
    if x.shape[-1] == 2:
        valid = x[:, 0] * 0 == 0
    elif x.shape[-1] == 3:
        valid = x[:, 2] > 0
    else:
        assert 0
    v = x[valid]
    if len(v) < 2:
        return x[0][0] * 0
    # TODO(xiao): plus one?
    w = max(v[:, 0]) - min(v[:, 0])
    h = max(v[:, 1]) - min(v[:, 1])
    return w * w + h * h


def cal_area_2_torch(v):
    # TODO(xiao): plus one?
    w = torch.max(v[:, :, 0], -1)[0] - torch.min(v[:, :, 0], -1)[0]
    h = torch.max(v[:, :, 1], -1)[0] - torch.min(v[:, :, 1], -1)[0]
    return w * w + h * h


class KMeans():
    def __init__(self, K, X):
        self.K = K
        self.X = X
        self.N = len(X)
        self.mu = None
        self.clusters = None
        self.method = None

    def _cluster_points(self):
        mu = copy.deepcopy(self.mu)
        clusters = {}
        for x in self.X:
            bestmukey = min([(i[0], dist_kpt(x, mu[i[0]])) \
                             for i in enumerate(mu)], key=lambda t: t[1])[0]
            try:
                clusters[bestmukey].append(x)
            except KeyError:
                clusters[bestmukey] = [x]
        self.clusters = clusters

    def _reevaluate_centers(self):
        clusters = copy.deepcopy(self.clusters)
        newmu = []
        keys = sorted(self.clusters.keys())
        for k in keys:
            newmu.append(mean_kpt(clusters[k]))
        self.mu = newmu

    def _has_converged_kpt(self):
        oldmu = copy.deepcopy(self.oldmu)
        mu = copy.deepcopy(self.mu)
        if len(mu) == len(oldmu):
            while mu:
                a = mu[0]
                matched = False
                for m in range(0, len(oldmu)):
                    b = oldmu[m]
                    d = dist_kpt(a, b)
                    if d < KMEANS_DIST_THRE:
                        # find a match
                        matched = True
                        # remove matched pairs
                        del mu[0]
                        del oldmu[m]
                        break
                if not matched:
                    return False
            return True
        else:
            return False

    def find_centers(self, method='random'):
        self.method = method
        X = copy.deepcopy(self.X)
        K = copy.deepcopy(self.K)
        self.oldmu = random.sample(list(X), K)
        if method != '++':
            # Initialize to K random centers
            self.mu = random.sample(list(X), K)
        while not self._has_converged_kpt():
            self.oldmu = self.mu
            # Assign all points in X to clusters
            self._cluster_points()
            # Reevaluate centers
            self._reevaluate_centers()


class KPlusPlus(KMeans):
    def _dist_from_centers(self):
        cent = copy.deepcopy(self.mu)
        X = copy.deepcopy(self.X)
        D2 = np.array([min([dist_kpt(x, c) ** 2 for c in cent]) for x in X])
        self.D2 = D2

    def _choose_next_center(self):
        self.probs = self.D2 / self.D2.sum()
        self.cumprobs = self.probs.cumsum()
        r = random.random()
        ind = np.where(self.cumprobs >= r)[0][0]
        return (self.X[ind])

    def init_centers(self):
        self.mu = random.sample(list(self.X), 1)
        while len(self.mu) < self.K:
            self._dist_from_centers()
            self.mu.append(self._choose_next_center())


# data = [430.2521, 292.43698, 2.0, 460.5042, 233.61345, 2.0, 386.55463, 304.2017, 2.0, 576.4706, 210.08403, 2.0, 0.0,
#         0.0, 0.0, 620.1681, 336.13446, 2.0, 615.12604, 245.37816, 2.0, 532.77313, 487.39496, 2.0, 0.0, 0.0, 0.0,
#         391.59665, 640.3361, 2.0, 0.0, 0.0, 0.0, 1003.3613, 522.6891, 2.0, 1003.3613, 462.18488, 2.0, 0.0, 0.0, 0.0,
#         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 504.28638, 360.56335, 2.0, 524.94366, 345.5399, 2.0, 481.75122,
#         339.9061, 2.0, 539.96716, 364.31924, 2.0, 447.94836, 354.92957, 2.0, 568.13617, 478.87323, 2.0, 378.46478,
#         452.58215, 2.0, 579.40375, 647.8873, 2.0, 325.88263, 550.23474, 1.0, 575.6479, 758.6854, 2.0, 0.0, 0.0, 0.0,
#         521.1878, 773.7089, 2.0, 412.26764, 768.0751, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
#         324.0047, 486.38495, 2.0, 339.0282, 467.60562, 2.0, 297.71362, 465.7277, 2.0, 0.0, 0.0, 0.0, 248.88733,
#         482.6291, 2.0, 370.95306, 610.3286, 2.0, 198.1831, 600.93896, 2.0, 0.0, 0.0, 0.0, 141.8451, 798.1221, 1.0,
#         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
#         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
#         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
#         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# data = np.asarray(data)
# data = data.reshape((-1, 17, 3))
# data = data[0:3, :, :]
# kplusplus = KPlusPlus(2, data)
# kplusplus.init_centers()
# kplusplus.find_centers(method='++')
# print(kplusplus.mu)
# print(kplusplus.clusters)


def cal_coco_pose_kmeans(res_roots, num_cluster, keep_pose_not_less_than=KEEP_POSE_NOT_LESS_THAN, file_name=""):
    # load poses
    res = []
    for n_folder in range(0, len(res_roots)):
        root_folder = res_roots[n_folder]
        for file in os.listdir(root_folder):
            if file.endswith(".npy"):
                res.append(np.load(root_folder + file))
    res = np.concatenate(res, axis=0)
    # res = res[0:1000]

    # process vis
    vis = res[:, :, 2]
    vis[vis >= 0.5] = 1
    vis[vis < 0.5] = 0
    vis_sum = np.sum(vis, axis=1)

    # filters
    has_left_shoulder = vis[:, LEFT_SHOULDER_ID] > 0.5
    has_right_shoulder = vis[:, RIGHT_SHOULDER_ID] > 0.5
    has_enough_kpt = vis_sum >= keep_pose_not_less_than
    condi = has_left_shoulder * has_right_shoulder * has_enough_kpt
    res = res[condi, :, :]

    # normalize
    mid_shoulder = (res[:, LEFT_SHOULDER_ID, 0:2] + res[:, RIGHT_SHOULDER_ID, 0:2]) * 0.5
    areas = []
    for n in range(0, len(res)):
        # TODO(xiao): other choice
        areas.append(cal_area_2(res[n]))
    areas_sqrt = np.sqrt(np.asarray(areas))
    for n in range(0, res.shape[1]):
        for m in range(0, 2):
            res[:, n, m] = (res[:, n, m] - mid_shoulder[:, m]) / areas_sqrt
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = res[..., 2] == 0
    res[inds] = 0

    # kmeans clustering
    kplusplus = KPlusPlus(num_cluster, res)
    kplusplus.init_centers()
    kplusplus.find_centers(method='++')

    if file_name:
        mmcv.dump(kplusplus, file_name)
    return kplusplus


# 'nose',
# 'left_eye',
# 'right_eye',
# 'left_ear',
# 'right_ear',
# 'left_shoulder',
# 'right_shoulder',
# 'left_elbow',
# 'right_elbow',
# 'left_wrist',
# 'right_wrist',
# 'left_hip',
# 'right_hip',
# 'left_knee',
# 'right_knee',
# 'left_ankle',
# 'right_ankle'
