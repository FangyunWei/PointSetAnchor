from __future__ import division

import numpy as np
import torch
import torch.nn as nn
from mmdet.ops import ModulatedDeformConv, DeformConv
from mmcv.cnn import normal_init
from mmdet.models.utils.norm import build_norm_layer
from ..utils import ConvModule, bias_init_with_prob

from mmdet.core import (TemplateGenerator, AnchorGenerator, template_target, force_fp32,
                        multi_apply, kpts_nms, pose2bbox_minmax, delta2bbox)
from ..builder import build_loss
from ..registry import HEADS
from mmdet.ops.nms import nms_wrapper

TEMPLATES = [0.012684601, -0.08933451, 1.0, 0.035536814, -0.10970051, 1.0, -0.0078309765, -0.11273975, 1.0, 0.06659797, -0.10087583, 1.0, -0.042694017, -0.108930714, 1.0, 0.09737194, 0.007991947, 1.0, -0.09737194, -0.007991949, 1.0, 0.14322653, 0.14098871, 1.0, -0.16548587, 0.11416945, 1.0, 0.12871487, 0.19020036, 1.0, -0.1469478, 0.16403662, 1.0, 0.038462676, 0.30842233, 1.0, -0.08971843, 0.30050713, 1.0, 0.043515097, 0.51921296, 1.0, -0.10788974, 0.51259565, 1.0, 0.022420978, 0.7422105, 1.0, -0.12596583, 0.7357682, 1.0,
             -0.029625744, -0.06555077, 1.0, -0.011077967, -0.09218792, 1.0, -0.05566849, -0.0839485, 1.0, 0.026786882, -0.098671325, 1.0, -0.085776635, -0.076187655, 1.0, 0.09570821, -0.023620017, 1.0, -0.0957082, 0.023620013, 1.0, 0.19405061, 0.056725252, 1.0, -0.136053, 0.1463625, 1.0, 0.19977416, 0.09758294, 1.0, -0.14284438, 0.1918328, 1.0, 0.13683, 0.24832788, 1.0, 0.012423863, 0.2715252, 1.0, 0.17028166, 0.40597346, 1.0, -0.009515348, 0.43237185, 1.0, 0.21985608, 0.6170284, 1.0, 0.033163197, 0.63931906, 1.0,
             -0.0034287644, -0.12739095, 1.0, 0.022156859, -0.15210669, 1.0, -0.029490544, -0.15155293, 1.0, 0.06186235, -0.13928963, 1.0, -0.0674362, -0.13784763, 1.0, 0.11579345, -0.0010220977, 1.0, -0.115793474, 0.0010220984, 1.0, 0.16543955, 0.16678892, 1.0, -0.16265573, 0.16963029, 1.0, 0.124735154, 0.24836159, 1.0, -0.12523934, 0.24837054, 1.0, 0.07678662, 0.32304317, 1.0, -0.07180694, 0.32377353, 1.0, 0.10069026, 0.48960128, 1.0, -0.09366286, 0.4903335, 1.0, 0.08333991, 0.7249029, 1.0, -0.07358884, 0.72530186, 1.0]

NEW_TEMPLATES = [-0.0034287644, -0.12739095, 1.0, 0.022156859, -0.15210669, 1.0, -0.029490544, -0.15155293, 1.0, 0.06186235, -0.13928963, 1.0, -0.0674362, -0.13784763, 1.0, 0.11579345, -0.0010220977, 1.0, -0.115793474, 0.0010220984, 1.0, 0.16543955, 0.16678892, 1.0, -0.16265573, 0.16963029, 1.0, 0.124735154, 0.24836159, 1.0, -0.12523934, 0.24837054, 1.0, 0.07678662, 0.32304317, 1.0, -0.07180694, 0.32377353, 1.0, 0.10069026, 0.48960128, 1.0, -0.09366286, 0.4903335, 1.0, 0.08333991, 0.7249029, 1.0, -0.07358884, 0.72530186, 1.0]

TEMPLATE_POINTS_NUM = 17


def delta2template(anchors,
                   anchors_scales,
                   deltas,
                   means,
                   stds,
                   max_shape,
                   use_out_scale
                   ):
    px = anchors[:, 0::2]
    py = anchors[:, 1::2]

    denorm_deltas = deltas
    dx = denorm_deltas[:, 0::2]
    dy = denorm_deltas[:, 1::2]

    if use_out_scale:
        xx = dx * anchors_scales + px
        yy = dy * anchors_scales + py
    else:
        pw = torch.max(px, -1)[0] - torch.min(px, -1)[0] + 1.0
        ph = torch.max(py, -1)[0] - torch.min(py, -1)[0] + 1.0
        xx = dx * pw.view(-1, 1) + px
        yy = dy * ph.view(-1, 1) + py

    if max_shape is not None:
        xx = xx.clamp(min=0, max=max_shape[1] - 1)
        yy = yy.clamp(min=0, max=max_shape[0] - 1)
    templates = torch.stack([xx, yy], dim=-1).view_as(deltas)
    return templates


def absorb_heatmap(poses, heat_pred, offset_pred, stride):
    pool = torch.nn.MaxPool2d(5, 1, 2)
    new_poses = poses.clone()
    maxm = pool(heat_pred[None,:,:,:])
    maxm = torch.eq(maxm, heat_pred).float()
    heat_pred = heat_pred * maxm
    
    heat_pred = heat_pred[0]
    num_joints, w = heat_pred.size(0), heat_pred.size(2)
    heat_pred = heat_pred.view(num_joints, -1)
    offset_pred = offset_pred.view(num_joints, 2, -1)
    val_k, ind = heat_pred.topk(30, dim=1)

    x = ind % w
    y = (ind / w).long()
    heats_ind = torch.stack((x, y), dim=2)  #17,30,2
    new_poses = new_poses.view(-1, num_joints, 2)

    for i in range(num_joints):
        offset_ind = offset_pred[i,:,ind[i]].permute(1,0)
        heat_ind = stride*(heats_ind[i].float()+offset_ind.float())
        kpt_ind = new_poses[:,i,:]
        kpts_heat_diff = kpt_ind[:,None,:] - heat_ind
        kpts_heat_diff.pow_(2)
        kpts_heat_diff = kpts_heat_diff.sum(2)
        kpts_heat_diff.sqrt_()
        keep_ind = torch.argmin(kpts_heat_diff, dim=1)
        
        new_poses[:,i,:] = heat_ind[keep_ind]

    new_poses = new_poses.view(-1, 2*num_joints)
    return new_poses


def get_dcn_setting(num_points):
    dcn_kernel = int(np.sqrt(num_points))
    dcn_pad = int((dcn_kernel - 1) / 2)
    assert dcn_kernel * dcn_kernel == num_points, \
        "The points number should be a square number."
    assert dcn_kernel % 2 == 1, \
        "The points number should be an odd square number."
    dcn_base = np.arange(-dcn_pad, dcn_pad + 1).astype(np.float64)
    dcn_base_y = np.repeat(dcn_base, dcn_kernel)
    dcn_base_x = np.tile(dcn_base, dcn_kernel)
    # Note: y first
    dcn_base_offset = np.stack([dcn_base_y, dcn_base_x], axis=1)
    dcn_base_offset = dcn_base_offset.reshape((-1))
    dcn_base_offset = torch.tensor(dcn_base_offset)
    return dcn_kernel, dcn_pad, dcn_base_offset


def permute_first_second(a_list):
    img_num = len(a_list)
    level_num = len(a_list[0])
    new_list = []
    for n_l in range(level_num):
        cts = []
        for n_i in range(img_num):
            cts.append(a_list[n_i][n_l])
        new_list.append(torch.stack(cts))
    return new_list


def visualize_template(t, name, image_size=800):
    from mmcv_custom.vis import vis_keypoints
    import cv2
    tmp = np.stack([t[:, 0], t[:, 1], t[:, 2], t[:, 2]])
    vis_img = np.ones((image_size, image_size, 3), dtype=np.uint8) * 255
    tmp[0, :] = tmp[0, :] * image_size / 2.0 + image_size / 2.0
    tmp[1, :] = tmp[1, :] * image_size / 2.0 + image_size / 2.0
    vis_img = vis_keypoints(vis_img, tmp, kp_thresh=0)
    cv2.imwrite('show_template_' + name + '.jpg', vis_img)
    return vis_img


def visualize_all_templates(ts):
    for n_temp in range(len(ts)):
        t = np.copy(ts[n_temp])
        visualize_template(t, "{}".format(n_temp))


def shift_templates_to_pelvis(_ts, index=[11, 12]):
    ts = np.copy(_ts)
    pelvis = (ts[:, index[0]:index[0]+1, 0:2] + ts[:, index[1]:index[1]+1, 0:2]) / 2.0
    ts[:, :, 0:2] = ts[:, :, 0:2] - pelvis
    return ts


def rotate_templates(_ts, rot):
    from mmcv_custom.image_transformation import get_rotation_matrix_2d, trans_points_2d, PI
    ts = np.copy(_ts)
    rot_radian = rot / 180.0 * PI
    mat = get_rotation_matrix_2d(rot_radian)
    for n_t in range(len(ts)):
        ts[n_t, :, 0], ts[n_t, :, 1] = trans_points_2d(ts[n_t, :, 0], ts[n_t, :, 1], mat)
    return ts

def get_ratio_from_pose_template(templates):
    ratios = []
    for t in templates:
        xs = t[:, 0]
        ys = t[:, 1]
        l = np.min(xs)
        r = np.max(xs)
        t = np.min(ys)
        b = np.max(ys)
        # ratio = h/w
        r = np.abs((b - t) / (r - l))
        ratios.append(r)
    return ratios

def get_templates_with_ratios(templates, ratios):
    assert len(templates) == 1
    new_templates = []
    for r in ratios:
        cur_templates = templates.copy()
        h_ratio = np.sqrt(r)
        w_ratio = 1 / h_ratio
        cur_templates[:,:,0] = cur_templates[:,:,0] * w_ratio
        cur_templates[:,:,1] = cur_templates[:,:,1] * h_ratio
        new_templates.append(cur_templates)
    new_templates = np.concatenate(new_templates, axis=0)
    return new_templates


def convert_points_to_tuple(points, x_scale, y_scale, x_offset, y_offset):
    tuple_list = []
    for i in range(len(points)//2):
        tuple_list.append((int(points[2*i] * x_scale + x_offset), int(points[2*i + 1] * y_scale + y_offset)))
    return tuple_list



def multiclass_nms_bbx_pose(multi_bboxes,
                            multi_bboxposes,
                            multi_scores,
                            multi_vises,
                            score_thr,
                            nms_cfg,
                            key_point_num,
                            bbx_point_num,
                            max_num=-1,
                            score_factors=None):
    """NMS for multi-class bboxes and poses.
    Args:
        multi_bboxes (Tensor): shape (n, #class*4)[cls aware] or (n, 4)[cls agnostic]
        multi_scores (Tensor): shape (n, #class), where the 0th column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.
        score_factors (Tensor): The factors multiplied to scores before
            applying NMS
    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 5) and (k, 1). Labels
            are 0-based.
    """
    num_classes = multi_scores.shape[1]
    bboxes, labels, bboxposes, vises = [], [], [], []
    nms_cfg_ = nms_cfg.copy()
    nms_type = nms_cfg_.pop('type', 'nms')
    nms_op = getattr(nms_wrapper, nms_type)
    for i in range(1, num_classes):
        cls_inds = multi_scores[:, i] > score_thr
        if not cls_inds.any():
            continue
        _bboxes = multi_bboxes[cls_inds, :] 
        _scores = multi_scores[cls_inds, i]
        _vises = multi_vises[cls_inds, :]
        _bboxposes = torch.cat([multi_bboxposes[cls_inds, :], _scores[:, None]], dim=1)   
        if score_factors is not None:
            _scores *= score_factors[cls_inds]
        cls_dets = torch.cat([_bboxes, _scores[:, None]], dim=1)
        cls_dets, nms_idx = nms_op(cls_dets, **nms_cfg_)
        cls_bboxposes = _bboxposes[nms_idx]
        cls_vises = _vises[nms_idx]
        cls_labels = multi_bboxes.new_full((cls_dets.shape[0], ),
                                            i - 1,
                                            dtype=torch.long)
        bboxes.append(cls_dets)
        bboxposes.append(cls_bboxposes)
        labels.append(cls_labels)
        vises.append(cls_vises)
    if bboxes:
        bboxes = torch.cat(bboxes)
        bboxposes = torch.cat(bboxposes)
        labels = torch.cat(labels)
        vises = torch.cat(vises)
        if bboxes.shape[0] > max_num:
            _, inds = bboxes[:, -1].sort(descending=True)
            inds = inds[:max_num]
            bboxes = bboxes[inds]
            bboxposes = bboxposes[inds]
            labels = labels[inds]
            vises = vises[inds]
    else:
        bboxes = multi_bboxes.new_zeros((0, 5))
        bboxposes = multi_bboxes.new_zeros((0, (key_point_num + bbx_point_num) * 2 + 1))
        labels = multi_bboxes.new_zeros((0, ), dtype=torch.long)
        vises = multi_bboxes.new_zeros((0, key_point_num))

    return bboxposes, labels, vises


@HEADS.register_module
class PointSetAnchorPoseHead(nn.Module):
    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 stacked_convs=4,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 # template/anchor
                 octave_base_scale=4,
                 scales_per_octave=3,
                 anchor_ratios=[0.5, 1.0, 2.0],
                 anchor_scales=[8, 16, 32],
                 anchor_strides=[4, 8, 16, 32, 64],
                 anchor_base_sizes=None,
                 target_means=(.0, .0, .0, .0),
                 target_stds=(1.0, 1.0, 1.0, 1.0),
                 anchor_rotations=[],
                 use_out_scale=False,
                 # template/anchor
                 # shape feature align
                 use_shape_index_feature=False,
                 fea_point_index=[0, 1, 2, 3, 4, 5, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 12, 13, 13, 14, 14, 15, 15, 16, 16],
                 modulated_dcn=True,
                 # shape feature align
                 # loss
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_reg=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
                 loss_bbox=dict(type='SmoothL1Loss', beta=0.11, loss_weight=1.0)
                 # loss
                 ):
        super(PointSetAnchorPoseHead, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.use_out_scale = use_out_scale
        self.norm_cfg = norm_cfg

        self.use_shape_index_feature = use_shape_index_feature
        self.modulated_dcn = modulated_dcn
        self.fea_point_index = fea_point_index
        self.dcn_kernel, self.dcn_pad, self.dcn_base_offset = get_dcn_setting(len(self.fea_point_index))

        self.anchor_scales_p = anchor_scales
        self.anchor_scales_b = [anchor_scales[i]/2 for i in range(len(anchor_scales))]
        self.anchor_strides = anchor_strides
        self.anchor_base_sizes = list(
            anchor_strides) if anchor_base_sizes is None else anchor_base_sizes
        self.target_means = target_means
        self.target_stds = target_stds

        self.templates = np.array(NEW_TEMPLATES)
        self.templates = self.templates.reshape(-1, TEMPLATE_POINTS_NUM, 3)
        self.templates = shift_templates_to_pelvis(self.templates)
        if anchor_rotations:
            rot_temps = []
            for rot in anchor_rotations:
                rot_temps.append(rotate_templates(self.templates, rot))
            self.templates = np.concatenate(rot_temps)
        self.anchor_generators = []
        # bbx anchor generators
        self.anchor_generators_bbx = []
        anchor_ratios = get_ratio_from_pose_template(self.templates)
        
        for anchor_base in self.anchor_base_sizes:
            self.anchor_generators.append(
                TemplateGenerator(anchor_base, self.anchor_scales_p, self.templates))
            self.anchor_generators_bbx.append(
                AnchorGenerator(anchor_base, self.anchor_scales_b, anchor_ratios))
        self.num_anchors = len(self.templates) * len(self.anchor_scales_p)

        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        self.sampling = loss_cls['type'] not in ['FocalLoss', 'GHMC']
        if self.use_sigmoid_cls:
            self.cls_out_channels = num_classes - 1
        else:
            self.cls_out_channels = num_classes
        if self.cls_out_channels <= 0:
            raise ValueError('num_classes={} is too small'.format(num_classes))
        
        self.loss_cls = build_loss(loss_cls)
        self.loss_reg = build_loss(loss_reg)
        self.loss_bbox = build_loss(loss_bbox)
        self.fp16_enabled = False

        self._init_layers()

    def _init_cls_and_reg_layers(self, num_anchors):
        conv_cls = nn.ModuleList()
        conv_reg = nn.ModuleList()
        conv_reg_bbx = nn.ModuleList()
        channels = self.feat_channels if self.use_shape_index_feature else self.in_channels
        for i in range(num_anchors):
            conv_cls.append(nn.Conv2d(channels, self.cls_out_channels, 1))
            conv_reg.append(nn.Conv2d(channels, 2 * TEMPLATE_POINTS_NUM, 1))
            conv_reg_bbx.append(nn.Conv2d(channels, 4, 1))
        return conv_cls, conv_reg, conv_reg_bbx

    def _init_cls_and_reg_pre_layers(self, stacked_convs):
        cls_convs = nn.ModuleList()
        reg_convs = nn.ModuleList()
        reg_convs_bbx = nn.ModuleList()
        for i in range(stacked_convs):
            cls_convs.append(
                ConvModule(
                    self.in_channels,
                    self.in_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=None,
                    norm_cfg=self.norm_cfg,
                    activation='relu',
                    inplace=False,
                ))
            reg_convs.append(
                ConvModule(
                    self.in_channels,
                    self.in_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=None,
                    norm_cfg=self.norm_cfg,
                    activation='relu',
                    inplace=False,
                ))
            reg_convs_bbx.append(
                ConvModule(
                    self.in_channels,
                    self.in_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=None,
                    norm_cfg=self.norm_cfg,
                    activation='relu',
                    inplace=False,
                ))
        return cls_convs, reg_convs, reg_convs_bbx

    def _init_shape_index_layers(self, num_anchors):
        shape_align_conv = nn.ModuleList()
        norm = nn.ModuleList()
        for i in range(num_anchors):
            if self.modulated_dcn:
                shape_align_conv.append(
                    ModulatedDeformConv(self.in_channels, self.feat_channels, self.dcn_kernel, stride=1,
                                        padding=self.dcn_pad, bias=False))
            else:
                shape_align_conv.append(
                    DeformConv(self.in_channels, self.feat_channels, self.dcn_kernel, stride=1,
                               padding=self.dcn_pad, bias=False))
            n_name, n = build_norm_layer(self.norm_cfg, self.feat_channels)
            norm.append(n)
        return shape_align_conv, norm

    def _init_shape_index_layers_pack(self):
        shape_align_conv_list = nn.ModuleList()
        norm_list = nn.ModuleList()
        shape_align_conv, norm = self._init_shape_index_layers(self.num_anchors)
        shape_align_conv_list.append(shape_align_conv)
        norm_list.append(norm)
        for i in range(len(self.anchor_strides) - 1):
            shape_align_conv_list.append(shape_align_conv)
            norm_list.append(norm)
        return shape_align_conv_list, norm_list

    def _init_layers(self):
        self.relu = nn.ReLU(inplace=False)

        self.conv_cls_list = nn.ModuleList()
        self.conv_reg_list = nn.ModuleList()
        self.conv_reg_bbx_list = nn.ModuleList()
        conv_cls, conv_reg, conv_reg_bbx = self._init_cls_and_reg_layers(self.num_anchors)
        self.conv_cls_list.append(conv_cls)
        self.conv_reg_list.append(conv_reg)
        self.conv_reg_bbx_list.append(conv_reg_bbx)
        for i in range(len(self.anchor_strides) - 1):
            self.conv_cls_list.append(conv_cls)
            self.conv_reg_list.append(conv_reg)
            self.conv_reg_bbx_list.append(conv_reg_bbx)

        self.pre_cls_convs, self.pre_reg_convs, self.pre_reg_bbx_convs = self._init_cls_and_reg_pre_layers(self.stacked_convs)

        if self.use_shape_index_feature:
            self.shape_align_conv_list, self.norm_list = self._init_shape_index_layers_pack()
            self.shape_align_conv_list_cls, self.norm_list_cls = self._init_shape_index_layers_pack()
            self.shape_align_conv_list_reg, self.norm_list_reg = self._init_shape_index_layers_pack()
            self.shape_align_conv_list_reg_bbx, self.norm_list_reg_bbx = self._init_shape_index_layers_pack()

    def init_weights(self):
        for i in range(len(self.conv_cls_list)):
            for j in range(len(self.conv_cls_list[i])):
                normal_init(self.conv_cls_list[i][j], std=0.01)
                normal_init(self.conv_reg_list[i][j], std=0.01)
                normal_init(self.conv_reg_bbx_list[i][j], std=0.01)

        for m in self.pre_cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.pre_reg_convs:
            normal_init(m.conv, std=0.01)
        for m in self.pre_reg_bbx_convs:
            normal_init(m.conv, std=0.01)

        if self.use_shape_index_feature:
            for i in range(len(self.norm_list)):
                for j in range(len(self.norm_list[i])):
                    nn.init.constant_(self.norm_list[i][j].weight, 1)
                    nn.init.constant_(self.norm_list[i][j].bias, 0)
                    nn.init.constant_(self.norm_list_cls[i][j].weight, 1)
                    nn.init.constant_(self.norm_list_cls[i][j].bias, 0)
                    nn.init.constant_(self.norm_list_reg[i][j].weight, 1)
                    nn.init.constant_(self.norm_list_reg[i][j].bias, 0)
                    nn.init.constant_(self.norm_list_reg_bbx[i][j].weight, 1)
                    nn.init.constant_(self.norm_list_reg_bbx[i][j].bias, 0)

    def forward_single(self, x, anchors_, valid_flags_, anchors_zero_, index):
        if self.use_shape_index_feature:
            anchors = anchors_.clone()
            valid_flags = valid_flags_.clone()
            anchors_zero = anchors_zero_.clone()
            anchors_dcn = anchors - anchors_zero
            anchors_dcn = anchors_dcn.view((x.shape[0], x.shape[-2], x.shape[-1], -1, TEMPLATE_POINTS_NUM, 2))
            valid_flags = valid_flags.view((x.shape[0], x.shape[-2], x.shape[-1], -1, 1))

            cls_feat = x
            reg_feat = x
            reg_bbx_feat = x
            for cls_conv in self.pre_cls_convs:
                cls_feat = cls_conv(cls_feat)
            for reg_conv in self.pre_reg_convs:
                reg_feat = reg_conv(reg_feat)
            for reg_bbx_conv in self.pre_reg_bbx_convs:
                reg_bbx_feat = reg_bbx_conv(reg_bbx_feat)
            fea_cls = self.shape_align_single(
                cls_feat, anchors_dcn, valid_flags, index, self.norm_list_cls, self.shape_align_conv_list_cls)
            fea_reg = self.shape_align_single(
                reg_feat, anchors_dcn, valid_flags, index, self.norm_list_reg, self.shape_align_conv_list_reg)
            fea_reg_bbx = self.shape_align_single(
                reg_bbx_feat, anchors_dcn, valid_flags, index, self.norm_list_reg_bbx, self.shape_align_conv_list_reg_bbx)
        else:
            cls_feat = x
            reg_feat = x
            reg_bbx_feat = x
            for cls_conv in self.pre_cls_convs:
                cls_feat = cls_conv(cls_feat)
            for reg_conv in self.pre_reg_convs:
                reg_feat = reg_conv(reg_feat)
            for reg_bbx_conv in self.pre_reg_bbx_convs:
                reg_bbx_feat = reg_bbx_conv(reg_bbx_feat)
            fea_cls = []
            for n_shape in range(self.num_anchors):
                fea_cls.append(cls_feat)
            fea_reg = []
            for n_shape in range(self.num_anchors):
                fea_reg.append(reg_feat)
            fea_reg_bbx = []
            for n_shape in range(self.num_anchors):
                fea_reg_bbx.append(reg_bbx_feat)

        cls_score = []
        reg_pred = []
        reg_bbx_pred = []
        for n_shape in range(self.num_anchors):
            cls_score.append(self.conv_cls_list[index][n_shape](fea_cls[n_shape]))
            reg_pred.append(self.conv_reg_list[index][n_shape](fea_reg[n_shape]))
            reg_bbx_pred.append(self.conv_reg_bbx_list[index][n_shape](fea_reg_bbx[n_shape]))

        cls_score = torch.cat(cls_score, dim=1)
        reg_pred = torch.cat(reg_pred, dim=1)
        reg_bbx_pred = torch.cat(reg_bbx_pred, dim=1)
        return cls_score, reg_pred, reg_bbx_pred

    def shape_align_single(self, x, anchor, valid_flag, index, norm_list, shape_align_conv_list):
        stride = self.anchor_strides[index]
        offset = anchor / stride
        offset = offset[:, :, :, :, self.fea_point_index, :]
        offset_x = offset[:, :, :, :, :, 0]
        offset_y = offset[:, :, :, :, :, 1]
        offset = torch.stack([offset_y, offset_x], dim=5)
        num_shape = offset.shape[3]
        dcn_base_offset = self.dcn_base_offset.type_as(x)

        aligned_fea = []
        for n_shape in range(num_shape):
            off = offset[:, :, :, n_shape, :, :].view(
                (offset.shape[0], offset.shape[1], offset.shape[2], -1)).clone()
            off = off - dcn_base_offset
            off = off.permute(0, 3, 1, 2)
            _off = off.clone()
            mask = _off.new_ones(_off.shape)
            if self.modulated_dcn:
                a_fea = self.relu(norm_list[index][n_shape](
                    shape_align_conv_list[index][n_shape](x, _off, mask)))
            else:
                a_fea = self.relu(norm_list[index][n_shape](
                    shape_align_conv_list[index][n_shape](x, _off)))
            aligned_fea.append(a_fea)
        return aligned_fea

    def forward(self, feats, anchor_list, valid_flag_list, anchor_zero_list):
        anchor_list = permute_first_second(anchor_list)
        valid_flag_list = permute_first_second(valid_flag_list)
        anchor_zero_list = permute_first_second(anchor_zero_list)
        return multi_apply(self.forward_single, feats, anchor_list, valid_flag_list, anchor_zero_list,
                           range(len(feats)))

    def get_anchors(self, featmap_sizes, img_metas, device='cuda'):
        num_imgs = len(img_metas)
        num_levels = len(featmap_sizes)

        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        multi_level_anchors = []
        multi_level_bbx_anchors = []
        multi_level_anchors_zero = []
        multi_level_anchors_scales = []
        for i in range(num_levels):
            anchors, zero_anchors, anchors_scales = self.anchor_generators[i].grid_anchors(
                featmap_sizes[i], self.anchor_strides[i], device=device)
            multi_level_anchors.append(anchors)
            bbx_anchors = self.anchor_generators_bbx[i].grid_anchors(
                featmap_sizes[i], self.anchor_strides[i], device=device)

            multi_level_bbx_anchors.append(bbx_anchors)
            multi_level_anchors_zero.append(zero_anchors)
            multi_level_anchors_scales.append(anchors_scales)
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]
        anchor_bbx_list = [multi_level_bbx_anchors for _ in range(num_imgs)]
        anchor_zero_list = [multi_level_anchors_zero for _ in range(num_imgs)]
        anchor_scale_list = [multi_level_anchors_scales for _ in range(num_imgs)]

        # for each image, we compute valid flags of multi level anchors
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = []
            for i in range(num_levels):
                anchor_stride = self.anchor_strides[i]
                feat_h, feat_w = featmap_sizes[i]
                h, w, _ = img_meta['pad_shape']
                valid_feat_h = min(int(np.ceil(h / anchor_stride)), feat_h)
                valid_feat_w = min(int(np.ceil(w / anchor_stride)), feat_w)
                flags = self.anchor_generators[i].valid_flags(
                    (feat_h, feat_w), (valid_feat_h, valid_feat_w),
                    device=device)
                multi_level_flags.append(flags)
            valid_flag_list.append(multi_level_flags)

        return anchor_list, anchor_bbx_list, valid_flag_list, anchor_zero_list, anchor_scale_list

    def loss_single(self, cls_score, reg_pred, reg_bbx_pred, labels, label_weights,
                    reg_targets, reg_weights, reg_bbx_targets, reg_bbx_weights, anchors, anchor_scales, num_total_samples, cfg):
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples)
        # regression loss
        reg_targets = reg_targets.reshape(-1, 2 * TEMPLATE_POINTS_NUM)
        reg_weights = reg_weights.reshape(-1, 2 * TEMPLATE_POINTS_NUM)
        reg_pred = reg_pred.permute(0, 2, 3, 1).reshape(-1, 2 * TEMPLATE_POINTS_NUM)
        loss_reg = self.loss_reg(
            reg_pred,
            reg_targets,
            reg_weights,
            avg_factor=num_total_samples)
        # bbx regression loss
        reg_bbx_targets = reg_bbx_targets.reshape(-1, 4)
        reg_bbx_weights = reg_bbx_weights.reshape(-1, 4)
        reg_bbx_pred = reg_bbx_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        loss_bbx = self.loss_bbox(
            reg_bbx_pred,
            reg_bbx_targets,
            reg_bbx_weights,
            avg_factor=num_total_samples)
        return loss_cls, loss_reg, loss_bbx

    @force_fp32(apply_to=('cls_scores', 'reg_preds', 'reg_bbx_preds'))
    def loss(self,
             cls_scores,
             reg_preds,
             reg_bbx_preds,
             gt_labels,
             img_metas,
             cfg,
             gt_keypoints=None,
             gt_bboxes=None,
             out_anchor_list=None,
             out_anchor_bbx_list=None,
             out_valid_flag_list=None,
             out_anchor_scale_list=None,
             gt_bboxes_ignore=None
             ):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == len(self.anchor_generators)

        device = cls_scores[0].device

        if out_anchor_list is None:
            anchor_list, anchor_bbx_list, valid_flag_list, anchor_zero_list, anchor_scale_list = self.get_anchors(
                featmap_sizes, img_metas, device=device)
        else:
            anchor_list = out_anchor_list
            anchor_bbx_list = out_anchor_bbx_list
            valid_flag_list = out_valid_flag_list
            anchor_scale_list = out_anchor_scale_list

        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1

        anchor_infos = {
            'featmap_sizes': featmap_sizes,
            'point_anchor_num': self.anchor_generators[0].base_anchors.shape[0],
            'point_anchor_dim': self.anchor_generators[0].base_anchors.shape[1]
        }

        cls_reg_targets = template_target(
            anchor_list,
            anchor_bbx_list,
            anchor_scale_list,
            valid_flag_list,
            gt_keypoints,
            gt_bboxes,
            gt_bboxes_ignore,
            img_metas,
            self.target_means,
            self.target_stds,
            anchor_infos,
            self.use_out_scale,
            cfg,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
            sampling=self.sampling)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, reg_targets_list, reg_weights_list,
         reg_bbx_targets_list, reg_bbx_weights_list,
         num_total_pos, num_total_neg, num_pos_list, num_neg_list) = cls_reg_targets
        new_anchor_list = []
        new_anchor_scale_list = []
        sum_length = 0
        for i in range(len(reg_targets_list)):
            length = reg_targets_list[i].shape[1]
            new_anchor = torch.cat([anchor_list[j][sum_length:sum_length+length] for j in range(len(anchor_list))], dim=0)
            new_anchor_scale = torch.cat([anchor_scale_list[j][sum_length:sum_length+length] for j in range(len(anchor_scale_list))], dim=0)
            sum_length += length
            new_anchor_list.append(new_anchor)
            new_anchor_scale_list.append(new_anchor_scale)

        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)
        losses_cls, losses_reg, losses_bbx = multi_apply(
            self.loss_single,
            cls_scores,
            reg_preds,
            reg_bbx_preds,
            labels_list,
            label_weights_list,
            reg_targets_list,
            reg_weights_list,
            reg_bbx_targets_list,
            reg_bbx_weights_list,
            new_anchor_list,
            new_anchor_scale_list,
            num_total_samples=num_total_samples,
            cfg=cfg)

        loss_dict_all = {
            'loss_cls': losses_cls,
            'loss_reg': losses_reg,
            'loss_bbx': losses_bbx,
        }
        return loss_dict_all

    @force_fp32(apply_to=('cls_scores', 'reg_preds', 'heat_preds'))
    def get_bboxes(self, cls_scores, reg_preds, reg_bbx_preds, heat_preds, offset_preds, img_metas, cfg,
                   rescale=False, do_nms=True, use_heatmap=False, out_anchors=None, out_bbx_anchors=None,
                   out_anchors_scales=None, get_nextstage_anchor=False):
        assert len(cls_scores) == len(reg_preds) == len(reg_bbx_preds)
        num_levels = len(cls_scores)
        assert out_anchors is not None

        if out_anchors is None:
            device = cls_scores[0].device
            mlvl_anchors_ = [
                self.anchor_generators[i].grid_anchors(
                    cls_scores[i].size()[-2:],
                    self.anchor_strides[i],
                    device=device)[0] for i in range(num_levels)
            ]
            mlvl_anchors_scales_ = [
                self.anchor_generators[i].grid_anchors(
                    cls_scores[i].size()[-2:],
                    self.anchor_strides[i],
                    device=device)[2] for i in range(num_levels)
            ]
        else:
            mlvl_anchors_ = out_anchors
            mlvl_bbx_anchors_ = out_bbx_anchors
            mlvl_anchors_scales_ = out_anchors_scales

        if get_nextstage_anchor:
            anchor_list = []
            anchor_bbx_list = []

        result_list = []

        for img_id in range(len(img_metas)):
            if out_anchors is None:
                mlvl_anchors = mlvl_anchors_
                mlvl_anchors_scales = mlvl_anchors_scales_
            else:
                mlvl_anchors = mlvl_anchors_[img_id]
                mlvl_bbx_anchors = mlvl_bbx_anchors_[img_id]
                mlvl_anchors_scales = mlvl_anchors_scales_[img_id]
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            reg_pred_list = [
                reg_preds[i][img_id].detach() for i in range(num_levels)
            ]
            reg_bbx_pred_list = [
                reg_bbx_preds[i][img_id].detach() for i in range(num_levels)
            ]
            # use lowest heatmap
            heat_pred_list = [
                heat_preds[0][img_id].detach() for i in range(num_levels)
            ]
            # use lowest heatmap
            offset_pred_list = [
                offset_preds[0][img_id].detach() for i in range(num_levels)
            ]
            
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            use_predict_bbx = False
            proposals = self.get_bboxes_single(cls_score_list, reg_pred_list, reg_bbx_pred_list, heat_pred_list, offset_pred_list,
                                               mlvl_anchors, mlvl_bbx_anchors, mlvl_anchors_scales, img_shape,
                                               scale_factor, cfg, rescale, do_nms, use_heatmap, use_predict_bbx, get_nextstage_anchor)

            if get_nextstage_anchor:
                anchor_list.append(proposals[0])
                anchor_bbx_list.append(proposals[1])
            result_list.append(proposals)
        if get_nextstage_anchor:
            return anchor_list, anchor_bbx_list
        return result_list

    def get_bboxes_single(self,
                          cls_score_list,
                          reg_pred_list,
                          reg_bbx_pred_list,
                          heat_pred_list,
                          offset_pred_list,
                          mlvl_anchors,
                          mlvl_bbx_anchors,
                          mlvl_anchors_scales,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False,
                          do_nms=True,
                          use_heatmap=False,
                          use_predict_bbx=False,
                          get_nextstage_anchor=False
                          ):
        assert len(cls_score_list) == len(reg_pred_list) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_poses = []
        mlvl_areas = []
        mlvl_vis = []
        for cls_score, reg_pred, reg_bbx_pred, heat_pred, offset_pred, anchors, bbx_anchors, anchors_scales in zip(cls_score_list,
                                                reg_pred_list, reg_bbx_pred_list, heat_pred_list, offset_pred_list, mlvl_anchors, mlvl_bbx_anchors, mlvl_anchors_scales):
            assert cls_score.size()[-2:] == reg_pred.size()[-2:]

            #(3*1, h, w), (3*34, h, w), (17, h, w)
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)
            reg_pred = reg_pred.permute(1, 2, 0).reshape(-1, 2 * TEMPLATE_POINTS_NUM)
            reg_bbx_pred = reg_bbx_pred.permute(1, 2, 0).reshape(-1, 4)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre and not get_nextstage_anchor:
                if self.use_sigmoid_cls:
                    max_scores, _ = scores.max(dim=1)
                else:
                    max_scores, _ = scores[:, 1:].max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                bbx_anchors = bbx_anchors[topk_inds, :]
                anchors_scales = anchors_scales[topk_inds, :]
                reg_pred = reg_pred[topk_inds, :]
                reg_bbx_pred = reg_bbx_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                
            poses = delta2template(anchors, anchors_scales, reg_pred, self.target_means,
                                   self.target_stds, img_shape, self.use_out_scale)
            #(1000, 34)
            # bbx encoding
            bboxes = delta2bbox(bbx_anchors, reg_bbx_pred, [0, 0, 0, 0], [1, 1, 1, 1], img_shape)
            # clamp pose points, bbx has been clamped in delta2bbx function

            if use_heatmap:
                stride = int((img_shape[0] + 31)//32 * 32)/heat_pred.shape[1]
                poses = absorb_heatmap(poses, heat_pred, offset_pred, stride)
            
            xxx = []
            yyy = []
            for n in range(0, TEMPLATE_POINTS_NUM):
                xxx.append(poses[:, n * 2].clamp(min=0, max=img_shape[1]))
                yyy.append(poses[:, n * 2 + 1].clamp(min=0, max=img_shape[0]))
            poses = torch.stack([xxx[0], yyy[0]], dim=-1)
            for n in range(1, TEMPLATE_POINTS_NUM):
                poses = torch.cat([poses, torch.stack([xxx[n], yyy[n]], dim=-1)], dim=-1)

            if not use_predict_bbx:
                bboxes = pose2bbox_minmax(poses)
            w = bboxes[:, 2] - bboxes[:, 0]
            h = bboxes[:, 3] - bboxes[:, 1]
            area = w * h
            vis = area.new_ones((area.shape[0], TEMPLATE_POINTS_NUM))

            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_poses.append(poses)
            mlvl_areas.append(area)
            mlvl_vis.append(vis)

        if get_nextstage_anchor:
            return mlvl_poses, mlvl_bboxes

        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_poses = torch.cat(mlvl_poses)
        mlvl_areas = torch.cat(mlvl_areas)
        mlvl_vis = torch.cat(mlvl_vis)

        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
            mlvl_poses /= mlvl_poses.new_tensor(scale_factor)
            mlvl_areas /= mlvl_areas.new_tensor(scale_factor * scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        if self.use_sigmoid_cls:
            # Add a dummy background class to the front when using sigmoid
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            mlvl_scores = torch.cat([padding, mlvl_scores], dim=1)

        if do_nms:
            det_poses, det_labels, det_vises = kpts_nms(torch.cat([mlvl_bboxes, mlvl_poses], dim=-1), mlvl_scores,
                                                        mlvl_areas, mlvl_vis, cfg.score_thr, cfg.nms, cfg.max_per_img)
            return det_poses, det_labels
        else:
            return mlvl_bboxes, mlvl_poses, mlvl_scores, mlvl_areas, mlvl_vis
