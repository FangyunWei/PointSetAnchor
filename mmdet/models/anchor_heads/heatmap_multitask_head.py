from __future__ import division

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init
from ..registry import HEADS
from ..utils import bias_init_with_prob, ConvModule
from ..builder import build_loss
from mmdet.core import (multi_apply, pose2bbox_minmax)


@HEADS.register_module
class HeatmapMultitaskHead(nn.Module):

    def __init__(self,
                 stacked_heat_convs,
                 stacked_offset_convs=1,
                 in_channels=256,
                 feat_channels=256,
                 num_points=17,
                 lr_mult=1.0,
                 deconv_with_bias=False,
                 add_deconv=False,
                 use_heatmap=False,
                 loss_heatmap='',
                 bg_weight=1,
                 guassian_sigma=None,
                 deconv_num_layers=3,
                 deconv_num_filters=[256, 256, 256],
                 deconv_num_kernels=[4, 4, 4],
                 stride=[4],
                 loss_offset=dict(type='L1Loss', loss_weight=1),
                 separate_out_conv=False,
                 with_offset=False,
                 conv_cfg=None,
                 norm_cfg=None):
        super(HeatmapMultitaskHead, self).__init__()
        self.stacked_heat_convs = stacked_heat_convs
        self.stacked_offset_convs = stacked_offset_convs
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.num_points = num_points
        self.with_offset = with_offset
        self.loss_offset = build_loss(loss_offset)
        self.use_heatmap = use_heatmap
        self.loss_heatmap = loss_heatmap
        self.bg_weight = bg_weight
        self.guassian_sigma = guassian_sigma
        self.stride = stride
        self.lr_mult = lr_mult
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.inplanes = 256
        self.separate_out_conv = separate_out_conv
        self.add_deconv = add_deconv
        self.deconv_num_layers = deconv_num_layers
        self.deconv_num_filters = deconv_num_filters
        self.deconv_num_kernels = deconv_num_kernels
        self.deconv_with_bias = deconv_with_bias

        self._init_layers()
        self.init_weights()

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=0.1))
            layers.append(nn.ReLU(inplace=True))
            
            self.inplanes = planes

        return nn.Sequential(*layers)

    def _init_fea_layers_base(self, num_conv):
        convs = nn.ModuleList()
        for i in range(num_conv):
            chn = self.in_channels if i == 0 else self.feat_channels
            convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    inplace=False
                ))
        return convs

    def _init_out_layers_base(self, out_count):
        return nn.Conv2d(
            # in_channels=self.deconv_num_filters[-1],
            in_channels=self.feat_channels,
            out_channels=self.num_points * out_count,
            kernel_size=1,
            stride=1,
            padding=0
        )

    def _init_layers_base(self, num_conv, out_count):
        if self.separate_out_conv:
            fea_convs_list = nn.ModuleList()
            out_convs_list = nn.ModuleList()
            for i in range(len(self.stride)):
                fea_convs_list.append(self._init_fea_layers_base(num_conv))
                out_convs_list.append(self._init_out_layers_base(out_count))
            return fea_convs_list, out_convs_list
        else:
            fea_convs = self._init_fea_layers_base(num_conv)
            out_convs = self._init_out_layers_base(out_count)
            return fea_convs, out_convs

    def _init_layers(self):
        self.relu = nn.ReLU()

        self.heat_convs, self.hm_out_layer_convs = \
            self._init_layers_base(self.stacked_heat_convs, 1)

        if self.with_offset:
            self.offset_convs, self.offset_out_layer_convs = \
                self._init_layers_base(self.stacked_offset_convs, 2)
        
        self.deconv_layers = self._make_deconv_layer(
            num_layers=self.deconv_num_layers,
            num_filters=self.deconv_num_filters,
            num_kernels=self.deconv_num_kernels,
        )
        if self.with_offset:
            self.deconv_layers_offset = self._make_deconv_layer(
                num_layers=self.deconv_num_layers,
                num_filters=self.deconv_num_filters,
                num_kernels=self.deconv_num_kernels,
            )

    def _init_fea_weights_base(self, convs):
        for m in convs:
            normal_init(m.conv, std=0.01)
            m.conv.lr_mult = self.lr_mult

    def _init_out_weights_base(self, conv):
        bias_cls = bias_init_with_prob(0.01)
        for m in conv.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01, bias=bias_cls)

    def init_weights_base(self, fea_convs, out_convs):
        if self.separate_out_conv:
            for convs in fea_convs:
                self._init_fea_weights_base(convs)
            for convs in out_convs:
                self._init_out_weights_base(convs)
        else:
            self._init_fea_weights_base(fea_convs)
            self._init_out_weights_base(out_convs)

    def init_weights(self):
        self.init_weights_base(self.heat_convs, self.hm_out_layer_convs)
        if self.with_offset:
            self.init_weights_base(self.offset_convs, self.offset_out_layer_convs)

    def forward_single(self, feat, index=None):
        if self.stacked_heat_convs == 0:
            heat_feat = feat
        else:
            if self.separate_out_conv:
                heat_convs = self.heat_convs[index]
            else:
                heat_convs = self.heat_convs
            heat_feat = heat_convs[0](feat)
            for i in range(1, len(heat_convs)):
                heat_feat = heat_convs[i](heat_feat)
        if self.with_offset:
            if self.stacked_offset_convs == 0:
                offset_feat = feat
            else:
                if self.separate_out_conv:
                    offset_convs = self.offset_convs[index]
                else:
                    offset_convs = self.offset_convs
                offset_feat = offset_convs[0](feat)
                for i in range(1, len(offset_convs)):
                    offset_feat = offset_convs[i](offset_feat)

        if self.add_deconv:
            heat_feat = self.deconv_layers(heat_feat)
            if self.with_offset:
                offset_feat = self.deconv_layers_offset(offset_feat)

        if self.separate_out_conv:
            heat_pred = self.hm_out_layer_convs[index](self.relu(heat_feat))
        else:
            heat_pred = self.hm_out_layer(self.relu(heat_feat))

        if self.with_offset:
            if self.separate_out_conv:
                heat_offset = self.offset_out_layer_convs[index](self.relu(offset_feat))
            else:
                heat_offset = self.offset_layer(self.relu(offset_feat))
            del feat
            heat_pred = heat_pred.sigmoid().clamp(min=1e-4, max=1 - 1e-4)
            heat_offset = heat_offset.sigmoid()
            return heat_feat, heat_pred, heat_offset
        del feat
        heat_pred = heat_pred.sigmoid().clamp(min=1e-4, max=1 - 1e-4)
        return heat_feat, heat_pred, None

    def forward(self, feats):
        if isinstance(feats, tuple) or isinstance(feats, list):
            return multi_apply(self.forward_single, feats, range(len(self.stride)))
        return self.forward_single(feats)

    def get_target(self,
                   pred_heatmaps_batch,
                   gt_keypoints_list,
                   # gt_bboxes_list,
                   idx):
        downsample = self.stride[idx]
        gt_heatmaps_batch = []
        gt_offset_batch = []
        gt_offset_weight_batch = []
        for pred_heatmaps, gt_keypoints in zip(pred_heatmaps_batch, gt_keypoints_list):
            output_h, output_w = pred_heatmaps.shape[-2:]
            kpt_num = gt_keypoints.shape[1]
            keypoints_heat = gt_keypoints.new_zeros((kpt_num, output_h, output_w), dtype=torch.float32)
            if self.with_offset:
                keypoints_heat_offset = gt_keypoints.new_zeros((kpt_num * 2, output_h, output_w),
                                                               dtype=torch.float32)
                keypoints_heat_offset_weight = gt_keypoints.new_zeros((kpt_num * 2, output_h, output_w),
                                                                      dtype=torch.float32)
            for i in range(len(gt_keypoints)):
                keypoint = gt_keypoints[i].reshape(kpt_num, 3)
                keypoint_visable = keypoint[keypoint[:, 2] > 0]
                keypoint_visable = keypoint_visable[:, 0:2].reshape(1, -1)
                bbox = pose2bbox_minmax(keypoint_visable)[0] / downsample
                h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
                if self.guassian_sigma:
                    radius = 3*self.guassian_sigma
                else:
                    radius = gaussian_radius((torch.ceil(h), torch.ceil(w)))
                radius = max(0, int(radius))

                img_h = keypoints_heat.shape[-2] * downsample
                img_w = keypoints_heat.shape[-1] * downsample
                is_valid = (keypoint[:, 2] > 0) * (keypoint[:, 0] >= 0) * (keypoint[:, 0] < img_w) * (
                            keypoint[:, 1] >= 0) * (keypoint[:, 1] < img_h)

                kpts_int = (keypoint[:, :2] / downsample).to(torch.int32)
                kpts_offset = (keypoint[:, :2] % downsample) / downsample
                for j in range(kpt_num):
                    if is_valid[j]:
                        pt_int = kpts_int[j]
                        if self.with_offset:
                            pt_offset = kpts_offset[j]
                            x, y = int(pt_int[0]), int(pt_int[1])
                            keypoints_heat_offset[2 * j:2 * j + 2][:, y, x] = pt_offset
                            keypoints_heat_offset_weight[2 * j:2 * j + 2][:, y, x] = 1
                        draw_gaussian(keypoints_heat[j], pt_int, radius)
            gt_heatmaps_batch.append(keypoints_heat)
            if self.with_offset:
                gt_offset_batch.append(keypoints_heat_offset)
                gt_offset_weight_batch.append(keypoints_heat_offset_weight)
        gt_heatmaps_batch = torch.stack(gt_heatmaps_batch, 0)
        if self.with_offset:
            gt_offset_batch = torch.stack(gt_offset_batch, 0)
            gt_offset_weight_batch = torch.stack(gt_offset_weight_batch, 0)
            return gt_heatmaps_batch, pred_heatmaps_batch, gt_offset_batch, gt_offset_weight_batch
        return gt_heatmaps_batch, pred_heatmaps_batch, None, None

    def loss_single(self, gt_heat, gt_offset, gt_offset_weight, pred_heat, offset):
        assert pred_heat.size()[-2:] == gt_heat.size()[-2:]
        if self.loss_heatmap == 'focal_loss':
            loss, num_pos = focal_loss(pred_heat, gt_heat, self.use_heatmap)
        elif self.loss_heatmap == 'tradeoff_l2_loss':
            loss, num_pos = tradeoff_l2_loss(pred_heat, gt_heat, self.bg_weight, self.use_heatmap)
        else:
            raise ValueError('Don not support this type of loss')

        if self.with_offset:
            offset_loss = self.loss_offset(
                offset,
                gt_offset,
                gt_offset_weight,
                avg_factor=max(torch.sum(gt_offset_weight > 0), 1)
            )
            return loss, num_pos, offset_loss
        else:
            return loss, num_pos, None

    def loss(self, gt_heat, gt_offset, gt_offset_weight, pred_heat, offset):
        losses = dict()
        assert len(pred_heat) == len(gt_heat)
        heat_loss, pos_num, offset_loss = multi_apply(self.loss_single, gt_heat, gt_offset, gt_offset_weight, pred_heat,
                                                      offset)
        losses['heat_loss'] = heat_loss
        if self.with_offset:
            losses['offset_loss'] = offset_loss
        losses['heat_pos_num'] = torch.tensor(pos_num)
        return losses


def focal_loss(pred, gt, use_heatmap=False):
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
      Arguments:
        pred (batch x c x h x w)
        gt_regr (batch x c x h x w)
    '''
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    if use_heatmap:
        neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds
    else:
        neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss, num_pos


def tradeoff_l2_loss(pred, gt, bg_weight, use_heatmap=False):
    ''' Modified l2 loss. 
      Arguments:
        pred (batch x c x h x w)
        gt_regr (batch x c x h x w)
    '''
    assert pred.size() == gt.size()
    pos_inds = gt.gt(0).float()
    neg_inds = gt.eq(0).float()
    mask = pos_inds+neg_inds*bg_weight
    loss = ((pred - gt)**2) * mask
    loss = loss.float().sum()

    num_pos = pos_inds.float().sum()
    return loss, num_pos


def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = torch.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = torch.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = torch.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    if r1 > r2:
        r1 = r2
    if r1 > r3:
        r1 = r3
    return r1


def draw_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
    gaussian = gaussian.type_as(heatmap)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = torch.exp(torch.from_numpy(-(x * x + y * y) / (2 * sigma * sigma)))
    h[h < torch.finfo(h.dtype).eps * h.max()] = 0
    return h






    
