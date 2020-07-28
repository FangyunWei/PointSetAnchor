import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from mmdet.core import (force_fp32, multi_apply, point_set_anchor_target,
                        delta2bbox, delta2points, distdelta2points,
                        get_corner_points_from_anchor_points)
from mmdet.ops import ModulatedDeformConvPack

from ..builder import build_loss
from ..registry import HEADS
from .anchor_head import AnchorHead
from ..utils import ConvModule, Scale, bias_init_with_prob, build_norm_layer
from mmdet.ops.nms import nms_wrapper
import numpy as np
import math
import time


@HEADS.register_module
class PointSetAnchorHead(AnchorHead):
    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 octave_base_scale=4,
                 scales_per_octave=3,
                 anchor_points_number=36,
                 mask_binary=True,
                 loss_mask=None,
                 loss_corner=None,
                 conv_cfg=None,
                 norm_cfg=None,
                 **kwargs):
        self.stacked_convs = stacked_convs
        self.octave_base_scale = octave_base_scale
        self.scales_per_octave = scales_per_octave
        self.anchor_points_number = anchor_points_number
        self.mask_out_channels = self.anchor_points_number
        self.corner_point_number = 4
        self.corner_out_channels = self.corner_point_number * 2
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        octave_scales = np.array(
            [2 ** (i / scales_per_octave) for i in range(scales_per_octave)])
        anchor_scales = octave_scales * octave_base_scale
        super(PointSetAnchorHead, self).__init__(
            num_classes, in_channels, anchor_scales=anchor_scales, **kwargs)
        self.loss_mask = build_loss(loss_mask) if loss_mask is not None else None
        self.loss_corner = build_loss(loss_corner) if loss_corner is not None else None
        self.loss_mask_type = loss_mask.type
        self.mask_binary = mask_binary


    def _init_layers(self):
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.mask_corner_convs = nn.ModuleList()

        num_point_set_anchors = self.num_anchors
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.mask_corner_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))

        self.anchor_points_cls = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * self.cls_out_channels,
            3,
            padding=1)
        self.anchor_points_reg = nn.Conv2d(
            self.feat_channels, self.num_anchors * 4, 3, padding=1)
        self.anchor_points_mask = nn.Conv2d(
            self.feat_channels,
            num_point_set_anchors * self.mask_out_channels,
            3, padding=1)
        self.anchor_points_corner = nn.Conv2d(
            self.feat_channels,
            num_point_set_anchors * self.corner_out_channels,
            3, padding=1)

    def init_weights(self):
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        for m in self.mask_corner_convs:
            normal_init(m.conv, std=0.01)

        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.anchor_points_cls, std=0.01, bias=bias_cls)
        normal_init(self.anchor_points_reg, std=0.01)
        normal_init(self.anchor_points_mask, std=0.01)
        normal_init(self.anchor_points_corner, std=0.01)

    def forward_single(self, x):
        cls_feat = x
        reg_feat = x
        mask_corner_feat = x

        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        for mask_corner_conv in self.mask_corner_convs:
            mask_corner_feat = mask_corner_conv(mask_corner_feat)

        cls_score = self.anchor_points_cls(cls_feat)
        bbox_pred = self.anchor_points_reg(reg_feat)
        mask_pred = self.anchor_points_mask(mask_corner_feat)
        corner_pred = self.anchor_points_corner(mask_corner_feat)

        return cls_score, bbox_pred, corner_pred, mask_pred


    def generate_anchor_points(self,
                               anchors,
                               anchor_points_num,
                               device):
        '''

        :param anchors: anchors of single level, Tensor(n, 4)
        :param anchor_points_num: template point number of each level
        :param device: device
        :return: generated template points,
                Tensor(n, anchor_points_num x 2)
        '''
        n = anchors.shape[0]
        anchor_points = torch.zeros((n, anchor_points_num, 2),
                                      device=device)
        # template point number in each side
        anchor_points_count_in_line = anchor_points.new_zeros((n, 4), dtype=torch.long)
        current_anchors = anchors.clone()
        center_x = (current_anchors[:, 0] + current_anchors[:, 2])/2
        center_y = (current_anchors[:, 1] + current_anchors[:, 3])/2
        current_anchors_w = current_anchors[:, 2] - current_anchors[:, 0] + 1
        current_anchors_h = current_anchors[:, 3] - current_anchors[:, 1] + 1
        anchors_x1 = center_x - (current_anchors_w - 1) / 2
        anchors_x2 = center_x + (current_anchors_w - 1) / 2
        anchors_y1 = center_y - (current_anchors_h - 1) / 2
        anchors_y2 = center_y + (current_anchors_h - 1) / 2
        anchors_w = anchors_x2 - anchors_x1 + 1
        anchors_h = anchors_y2 - anchors_y1 + 1
        # total template points num =
        # num_w_side*2 + num_extra + num_h_side*2
        num_wh_side = anchor_points_num // 2
        num_extra = anchor_points_num % 2
        unique_ws = anchors_w.unique(sorted=False)
        for unique_w in unique_ws:
            idx = (anchors_w == unique_w).nonzero().squeeze(1)
            current_x1 = anchors_x1[idx]
            current_x2 = anchors_x2[idx]
            current_y1 = anchors_y1[idx]
            current_y2 = anchors_y2[idx]
            current_w = anchors_w[idx[0]].item()
            current_h = anchors_h[idx[0]].item()
            current_wh = current_w + current_h
            # template points num for each w side
            num_w_side = int(current_w / current_wh * num_wh_side)
            # template points num for each h side
            num_h_side = num_wh_side - num_w_side
            # template points on top side
            top_invervals = torch.linspace(0, current_w - 1, num_w_side + num_extra + 2, device=device)[1:-1]
            top_x = current_x1.repeat(len(top_invervals), 1).permute(1, 0) + \
                    top_invervals.repeat(len(idx), 1)
            top_y = current_y1.repeat(len(top_invervals), 1).permute(1, 0)
            top_points = torch.stack([top_x, top_y], dim=2)
            # template points on right side
            right_invervals = torch.linspace(0, current_h - 1, num_h_side + 2, device=device)[1:-1]
            right_x = current_x2.repeat(len(right_invervals), 1).permute(1, 0)
            right_y = current_y1.repeat(len(right_invervals), 1).permute(1, 0) + \
                        right_invervals.repeat(len(idx), 1)
            right_points = torch.stack([right_x, right_y], dim=2)
            # template points on bottom side
            bottom_invervals = torch.linspace(current_w - 1, 0, num_w_side + 2, device=device)[1:-1]
            bottom_x = current_x1.repeat(len(bottom_invervals), 1).permute(1, 0) + \
                    bottom_invervals.repeat(len(idx), 1)
            bottom_y = current_y2.repeat(len(bottom_invervals), 1).permute(1, 0)
            bottom_points = torch.stack([bottom_x, bottom_y], dim=2)
            # template points on left side
            left_invervals = torch.linspace(current_h - 1, 0, num_h_side + 2, device=device)[1:-1]
            left_x = current_x1.repeat(len(left_invervals), 1).permute(1, 0)
            left_y = current_y1.repeat(len(left_invervals), 1).permute(1, 0) + \
                        left_invervals.repeat(len(idx), 1)
            left_points = torch.stack([left_x, left_y], dim=2)
            # template points should be in order
            all_points = torch.cat((top_points, right_points, bottom_points, left_points), dim=1)
            anchor_points[idx, :, 0: 2] = all_points
            anchor_points_count_in_line[idx, :] = anchor_points_count_in_line.new_tensor([num_w_side + num_extra,
                                                                                                num_h_side,
                                                                                                num_w_side,
                                                                                                num_h_side])
        t_n, t_tn, t_lv = anchor_points.shape
        anchor_points = anchor_points.view(t_n, t_tn * t_lv)
        return anchor_points, anchor_points_count_in_line


    def get_anchor_points(self, anchor_list,
                          anchor_points_num,
                          device='cuda'):
        '''

        :param anchor_list: different anchor bbxs of all images
        :param anchor_points_num: total template point number
        :param device: device
        :return: generated anchor points for all images
        '''
        num_imgs = len(anchor_list)
        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        anchors = anchor_list[0]
        num_levels = len(anchors)
        multi_level_anchor_points = []
        multi_level_anchor_points_count = []
        for i in range(num_levels):
            points, points_count_in_line = self.generate_anchor_points(anchors[i].clone(),
                                                                         anchor_points_num,
                                                                         device)
            multi_level_anchor_points.append(points)
            multi_level_anchor_points_count.append(points_count_in_line)
        anchor_points_list = [multi_level_anchor_points for _ in range(num_imgs)]
        anchor_points_count_list = [multi_level_anchor_points_count for _ in range(num_imgs)]
        return anchor_points_list, anchor_points_count_list

    def get_anchors(self, featmap_sizes, img_metas, device='cuda'):
        """Get anchors according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.
            device (torch.device | str): device for returned tensors

        Returns:
            tuple: anchors of each image, valid flags of each image
        """
        num_imgs = len(img_metas)
        num_levels = len(featmap_sizes)

        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        multi_level_anchors = []
        for i in range(num_levels):
            anchors = self.anchor_generators[i].grid_anchors(
                featmap_sizes[i], self.anchor_strides[i], device=device)
            multi_level_anchors.append(anchors)
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]

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

        return anchor_list, valid_flag_list

    def mask_points_merge(self, gt_masks_list):
        merged_gt_masks_list = []
        for gt_masks in gt_masks_list:
            merged_gt_mask = []
            for gt_mask in gt_masks:
                assert len(gt_mask) == 1
                merged_gt_mask.append(torch.Tensor(gt_mask[0]))
            merged_gt_masks_list.append(merged_gt_mask)
        return merged_gt_masks_list


    def loss_single(self, cls_score, bbox_pred, corner_pred, mask_pred, labels, label_weights,
                    bbox_targets, bbox_weights, mask_targets, mask_weights, mask_binary,
                    corner_targets, corner_weights, contour_targets, anchors, anchor_points,
                    num_total_samples, cfg):
        '''
        apply loss on each feature map
        '''
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples)
        # bbx regression loss
        bbox_targets = bbox_targets.reshape(-1, 4)
        bbox_weights = bbox_weights.reshape(-1, 4)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        loss_bbox = self.loss_bbox(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            avg_factor=num_total_samples)
        # corner regression loss
        corner_targets = corner_targets.reshape(-1, self.corner_out_channels)
        corner_weights = corner_weights.reshape(-1, self.corner_out_channels)
        corner_pred = corner_pred.permute(0, 2, 3, 1).reshape(-1, self.corner_out_channels)
        loss_corner = self.loss_corner(
            corner_pred,
            corner_targets,
            corner_weights,
            avg_factor=num_total_samples)
        # mask regression loss
        mask_targets = mask_targets.reshape(-1, self.mask_out_channels)
        mask_weights = mask_weights.reshape(-1, self.mask_out_channels)
        mask_binary = mask_binary.reshape(-1, self.mask_out_channels)
        mask_pred = mask_pred.permute(0, 2, 3, 1).reshape(-1, self.mask_out_channels)
        if self.mask_binary:
            mask_weights = mask_weights * mask_binary
        loss_mask = self.loss_mask(
            mask_pred,
            mask_targets,
            mask_weights,
            avg_factor=num_total_samples)
        return loss_cls, loss_bbox, loss_corner, loss_mask

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'corner_preds', 'mask_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             corner_preds,
             mask_preds,
             gt_bboxes,
             gt_labels,
             gt_masks,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == len(self.anchor_generators)

        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        anchor_points_list, anchor_points_count_list = \
            self.get_anchor_points(anchor_list,
                                     self.anchor_points_number,
                                     device=device)
        gt_masks = self.mask_points_merge(gt_masks)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = point_set_anchor_target(
            anchor_points_list,
            anchor_points_count_list,
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            self.target_means,
            self.target_stds,
            self.corner_point_number,
            cfg,
            featmap_sizes=featmap_sizes,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            gt_masks_list=gt_masks,
            label_channels=label_channels,
            sampling=self.sampling)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         mask_targets_list, mask_weights_list, mask_binary_list,
         corner_targets_list, corner_weights_list, contour_targets_list,
         anchor_list, anchor_points_list, num_total_pos, num_total_neg) \
            = cls_reg_targets
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)
        losses_cls, losses_bbox, losses_corner, losses_mask = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            corner_preds,
            mask_preds,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            mask_targets_list,
            mask_weights_list,
            mask_binary_list,
            corner_targets_list,
            corner_weights_list,
            contour_targets_list,
            anchor_list,
            anchor_points_list,
            num_total_samples=num_total_samples,
            cfg=cfg)
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox,
                    loss_corner=losses_corner, loss_mask=losses_mask)

    # inference
    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'mask_preds'))
    def get_inference_res(self, cls_scores, bbox_preds, corner_preds, mask_preds,
                          img_metas, cfg, rescale=None, nms=True):
        assert len(cls_scores) == len(bbox_preds) == len(corner_preds) == len(mask_preds)
        device = cls_scores[0].device
        num_levels = len(cls_scores)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        anchor_points_list, anchor_points_count_list =\
            self.get_anchor_points(anchor_list,
                                     self.anchor_points_number,
                                     device=device)
        mlvl_anchors = anchor_list[0]
        mlvl_anchor_points = anchor_points_list[0]
        mlvl_anchor_points_count = anchor_points_count_list[0]
        # proposals for len(img_metas) images
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            corner_pred_list = [
                corner_preds[i][img_id].detach() for i in range(num_levels)
            ]
            mask_pred_list = [
                mask_preds[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self.get_inference_res_single(cls_score_list, bbox_pred_list,
                                                      corner_pred_list, mask_pred_list,
                                                      mlvl_anchors, mlvl_anchor_points,
                                                      mlvl_anchor_points_count, img_shape,
                                                      scale_factor, cfg, rescale, nms)
            result_list.append(proposals)
        return result_list


    def get_inference_res_single(self,
                                 cls_score_list,
                                 bbox_pred_list,
                                 corner_pred_list,
                                 mask_pred_list,
                                 mlvl_anchors,
                                 mlvl_anchor_points,
                                 mlvl_anchor_points_count,
                                 img_shape,
                                 scale_factor,
                                 cfg,
                                 rescale=False,
                                 nms=True):
        """
        Transform outputs for a single batch item into labeled boxes.
        """
        assert len(cls_score_list) == len(bbox_pred_list) == \
               len(mask_pred_list) == len(corner_pred_list) ==\
               len(mlvl_anchors) == len(mlvl_anchor_points) ==\
               len(mlvl_anchor_points_count)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_masks = []
        for cls_score, bbox_pred, corner_pred, mask_pred, anchors, anchor_points, anchor_points_count in \
                zip(cls_score_list, bbox_pred_list, corner_pred_list, mask_pred_list,
                    mlvl_anchors, mlvl_anchor_points, mlvl_anchor_points_count):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:] == \
                   corner_pred.size()[-2:] == mask_pred.size()[-2:]
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            corner_pred = corner_pred.permute(1, 2, 0).reshape(-1, self.corner_out_channels)
            mask_pred = mask_pred.permute(1, 2, 0).reshape(-1, self.mask_out_channels)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                # Get maximum scores for foreground classes.
                if self.use_sigmoid_cls:
                    max_scores, _ = scores.max(dim=1)
                else:
                    max_scores, _ = scores[:, 1:].max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                anchor_points = anchor_points[topk_inds, :]
                anchor_points_count = anchor_points_count[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                corner_pred = corner_pred[topk_inds, :]
                mask_pred = mask_pred[topk_inds, :]
                scores = scores[topk_inds, :]
            # box decoding
            bboxes = delta2bbox(anchors, bbox_pred, self.target_means, self.target_stds, img_shape)
            # corner decoding
            corner_points = get_corner_points_from_anchor_points(anchor_points)
            corners = delta2points(corner_points, anchors, corner_pred, 0, 1)
            masks = distdelta2points(mask_pred, anchors, anchor_points, anchor_points_count, 0, 1)
            masks = self.point_ensemble(corners, masks, anchor_points_count, img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_masks.append(masks)
            mlvl_scores.append(scores)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_masks = torch.cat(mlvl_masks)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
            mlvl_masks /= mlvl_masks.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        if self.use_sigmoid_cls:
            # Add a dummy background class to the front when using sigmoid
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            mlvl_scores = torch.cat([padding, mlvl_scores], dim=1)
        if nms:
            det_bboxes, det_masks, det_labels = self.multiclass_nms_bbx_mask(mlvl_bboxes, mlvl_masks,
                                                                mlvl_scores, cfg.score_thr,
                                                                cfg.nms, cfg.max_per_img)
            return det_bboxes, det_masks, det_labels
        else:
            return mlvl_bboxes, mlvl_masks, mlvl_scores


    def point_ensemble(self, pred_corners, pred_masks, point_counter, max_shape=None):
        corners = pred_corners.clone()
        masks = pred_masks.clone()
        assert len(corners) == len(masks)
        if len(corners.shape) == 1:
            corners = corners.unsqueeze(-1)
        if len(masks.shape) == 1:
            masks = masks.unsqueeze(-1)
        if len(point_counter.shape) == 1:
            point_counter = point_counter.unsqueeze(-1)
        ensemble_res = masks.new_zeros((masks.shape[0], masks.shape[1] + corners.shape[1]))
        point_counter_index = torch.zeros_like(point_counter)
        tmp = point_counter.new_zeros(len(point_counter))
        for i in range(4):
            tmp += point_counter[:, i]
            point_counter_index[:, i] = tmp
        for l in point_counter[:, 0].unique():
            idx = (point_counter[:, 0] == l).nonzero().squeeze(-1)
            cur_corners = corners[idx, :]
            cur_masks = masks[idx, :]
            cur_point_counter_index = point_counter_index[idx, :][0]
            cur_ensemble_res = ensemble_res.new_zeros((len(idx), ensemble_res.shape[1]))
            lt_x = cur_corners[:, 0]
            lt_y = cur_corners[:, 1]
            rt_x = cur_corners[:, 2]
            rt_y = cur_corners[:, 3]
            rb_x = cur_corners[:, 4]
            rb_y = cur_corners[:, 5]
            lb_x = cur_corners[:, 6]
            lb_y = cur_corners[:, 7]
            # ensemble top points
            cur_ensemble_res[:, 0] = lt_x
            cur_ensemble_res[:, 1] = lt_y
            #limit x range of top points
            cur_masks_top_x = cur_masks[:, 0:cur_point_counter_index[0] * 2:2]
            cur_masks_top_y = cur_masks[:, 1:cur_point_counter_index[0] * 2:2]
            lt_x_idx = cur_masks_top_x < lt_x.unsqueeze(-1)
            cur_masks_top_x[lt_x_idx] = lt_x.unsqueeze(-1).repeat(1, cur_masks_top_x.shape[1])[lt_x_idx]
            cur_masks_top_y[lt_x_idx] = lt_y.unsqueeze(-1).repeat(1, cur_masks_top_y.shape[1])[lt_x_idx]
            rt_x_idx = cur_masks_top_x > rt_x.unsqueeze(-1)
            cur_masks_top_x[rt_x_idx] = rt_x.unsqueeze(-1).repeat(1, cur_masks_top_x.shape[1])[rt_x_idx]
            cur_masks_top_y[rt_x_idx] = rt_y.unsqueeze(-1).repeat(1, cur_masks_top_y.shape[1])[rt_x_idx]
            cur_ensemble_res[:, 2: cur_point_counter_index[0] *2 + 2:2] = cur_masks_top_x
            cur_ensemble_res[:, 3: cur_point_counter_index[0] *2 + 2:2] = cur_masks_top_y

            # ensemble right points
            cur_ensemble_res[:, 2 + cur_point_counter_index[0] * 2] = rt_x
            cur_ensemble_res[:, 3 + cur_point_counter_index[0] * 2] = rt_y
            #limit y range of right points
            cur_masks_right_x = cur_masks[:, cur_point_counter_index[0] * 2:cur_point_counter_index[1] * 2:2]
            cur_masks_right_y = cur_masks[:, cur_point_counter_index[0] * 2 + 1:cur_point_counter_index[1] * 2:2]
            rt_y_idx = cur_masks_right_y < rt_y.unsqueeze(-1)
            cur_masks_right_x[rt_y_idx] = rt_x.unsqueeze(-1).repeat(1, cur_masks_right_x.shape[1])[rt_y_idx]
            cur_masks_right_y[rt_y_idx] = rt_y.unsqueeze(-1).repeat(1, cur_masks_right_y.shape[1])[rt_y_idx]
            rb_y_idx = cur_masks_right_y > rb_y.unsqueeze(-1)
            cur_masks_right_x[rb_y_idx] = rb_x.unsqueeze(-1).repeat(1, cur_masks_right_x.shape[1])[rb_y_idx]
            cur_masks_right_y[rb_y_idx] = rb_y.unsqueeze(-1).repeat(1, cur_masks_right_y.shape[1])[rb_y_idx]
            cur_ensemble_res[:, 4 + cur_point_counter_index[0] * 2:4 + cur_point_counter_index[1] * 2:2] = cur_masks_right_x
            cur_ensemble_res[:, 5 + cur_point_counter_index[0] * 2:4 + cur_point_counter_index[1] * 2:2] = cur_masks_right_y

            # ensemble bottom points
            cur_ensemble_res[:, 4 + cur_point_counter_index[1] * 2] = rb_x
            cur_ensemble_res[:, 5 + cur_point_counter_index[1] * 2] = rb_y
            #limit x range of bottom points
            cur_masks_bottom_x = cur_masks[:, cur_point_counter_index[1] * 2:cur_point_counter_index[2] * 2:2]
            cur_masks_bottom_y = cur_masks[:, cur_point_counter_index[1] * 2 + 1:cur_point_counter_index[2] * 2:2]
            lb_x_idx = cur_masks_bottom_x < lb_x.unsqueeze(-1)
            cur_masks_bottom_x[lb_x_idx] = lb_x.unsqueeze(-1).repeat(1, cur_masks_bottom_x.shape[1])[lb_x_idx]
            cur_masks_bottom_y[lb_x_idx] = lb_y.unsqueeze(-1).repeat(1, cur_masks_bottom_y.shape[1])[lb_x_idx]
            rb_x_idx = cur_masks_bottom_x > rb_x.unsqueeze(-1)
            cur_masks_bottom_x[rb_x_idx] = rb_x.unsqueeze(-1).repeat(1, cur_masks_bottom_x.shape[1])[rb_x_idx]
            cur_masks_bottom_y[rb_x_idx] = rb_y.unsqueeze(-1).repeat(1, cur_masks_bottom_y.shape[1])[rb_x_idx]
            cur_ensemble_res[:, 6 + cur_point_counter_index[1] * 2:6 + cur_point_counter_index[2] * 2:2] = cur_masks_bottom_x
            cur_ensemble_res[:, 7 + cur_point_counter_index[1] * 2:6 + cur_point_counter_index[2] * 2:2] = cur_masks_bottom_y

            # ensemble left points
            cur_ensemble_res[:, 6 + cur_point_counter_index[2] * 2] = lb_x
            cur_ensemble_res[:, 7 + cur_point_counter_index[2] * 2] = lb_y
            #limit y range of left points
            cur_masks_left_x = cur_masks[:, cur_point_counter_index[2] * 2:cur_point_counter_index[3] * 2:2]
            cur_masks_left_y = cur_masks[:, cur_point_counter_index[2] * 2 + 1:cur_point_counter_index[3] * 2:2]
            lt_y_idx = cur_masks_left_y < lt_y.unsqueeze(-1)
            cur_masks_left_x[lt_y_idx] = lt_x.unsqueeze(-1).repeat(1, cur_masks_left_x.shape[1])[lt_y_idx]
            cur_masks_left_y[lt_y_idx] = lt_y.unsqueeze(-1).repeat(1, cur_masks_left_y.shape[1])[lt_y_idx]
            lb_y_idx = cur_masks_left_y > lb_y.unsqueeze(-1)
            cur_masks_left_x[lb_y_idx] = lb_x.unsqueeze(-1).repeat(1, cur_masks_left_x.shape[1])[lb_y_idx]
            cur_masks_left_y[lb_y_idx] = lb_y.unsqueeze(-1).repeat(1, cur_masks_left_y.shape[1])[lb_y_idx]
            cur_ensemble_res[:, 8 + cur_point_counter_index[2] * 2:8 + cur_point_counter_index[3] * 2:2] = cur_masks_left_x
            cur_ensemble_res[:, 9 + cur_point_counter_index[2] * 2:8 + cur_point_counter_index[3] * 2:2] = cur_masks_left_y

            ensemble_res[idx, :] = cur_ensemble_res


        if max_shape is not None:
            ensemble_res_x = ensemble_res[:, 0::2].clamp(min=0, max=max_shape[1] - 1)
            ensemble_res_y = ensemble_res[:, 1::2].clamp(min=0, max=max_shape[0] - 1)
            ensemble_res[:, 0::2] = ensemble_res_x
            ensemble_res[:, 1::2] = ensemble_res_y
        return ensemble_res



    def multiclass_nms_bbx_mask(self,
                                multi_bboxes,
                                multi_masks,
                                multi_scores,
                                score_thr,
                                nms_cfg,
                                max_num=-1,
                                score_factors=None):
        """NMS for multi-class bboxes and masks.

        Args:
            multi_bboxes (Tensor): shape (n, #class*4)[cls aware] or (n, 4)[cls agnostic]
            multi_masks (Tensor): shape (n, #class*templatenumber*2)[cls aware]
                                    or (n, templatenumber*2)[cls agnostic]
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
        bboxes, labels, masks = [], [], []
        nms_cfg_ = nms_cfg.copy()
        nms_type = nms_cfg_.pop('type', 'nms')
        nms_op = getattr(nms_wrapper, nms_type)
        for i in range(1, num_classes):
            cls_inds = multi_scores[:, i] > score_thr
            if not cls_inds.any():
                continue
            # get bboxes and scores of this class
            if multi_bboxes.shape[1] == 4:
                _bboxes = multi_bboxes[cls_inds, :]
            else:
                _bboxes = multi_bboxes[cls_inds, i * 4:(i + 1) * 4]
            _mask_len = self.mask_out_channels * 2 + self.corner_out_channels
            if multi_masks.shape[1] == _mask_len:
                _masks = multi_masks[cls_inds, :] #cls aware
            else:
                _masks = multi_masks[cls_inds, i * _mask_len:(i + 1) * _mask_len] #cls agnostic
                raise NotImplementedError
            _scores = multi_scores[cls_inds, i]
            if score_factors is not None:
                _scores *= score_factors[cls_inds]
            cls_dets = torch.cat([_bboxes, _scores[:, None]], dim=1)
            cls_dets, nms_idx = nms_op(cls_dets, **nms_cfg_)
            cls_masks = _masks[nms_idx]
            cls_labels = multi_bboxes.new_full((cls_dets.shape[0], ),
                                               i - 1,
                                               dtype=torch.long)
            bboxes.append(cls_dets)
            masks.append(cls_masks)
            labels.append(cls_labels)
        if bboxes:
            bboxes = torch.cat(bboxes)
            masks = torch.cat(masks)
            labels = torch.cat(labels)
            if bboxes.shape[0] > max_num:
                _, inds = bboxes[:, -1].sort(descending=True)
                inds = inds[:max_num]
                bboxes = bboxes[inds]
                masks = masks[inds]
                labels = labels[inds]
        else:
            bboxes = multi_bboxes.new_zeros((0, 5))
            masks = multi_bboxes.new_zeros((0, self.mask_out_channels))
            labels = multi_bboxes.new_zeros((0, ), dtype=torch.long)

        return bboxes, masks, labels
