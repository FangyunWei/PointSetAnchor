import torch
import numpy as np

# from ..geometry import bbox_overlaps
from scnn.kmeans.kmeans import cal_area_2, cal_area_2_torch
from mmdet.ops.nms.oks_nms_py import oks_iou_tensor
from .assign_result import AssignResult
from .base_assigner import BaseAssigner


def kpt_overlaps(gt_keypoints, templates, mode='iou'):
    """Calculate overlap between two set of bboxes.

    Args:
        bboxes1 (Tensor): shape (m, 4)
        bboxes2 (Tensor): shape (n, 4), if is_aligned is ``True``, then m and n
            must be equal.
        mode (str): "iou" (intersection over union) or iof (intersection over
            foreground).

    Returns:
        ious(Tensor): shape (m, n) if is_aligned == False else shape (m, 1)
    """

    assert mode in ['iou', 'iof']

    templates = templates.view(templates.shape[0], -1, 2)
    temp_vis = templates.new_ones((templates.shape[0], templates.shape[1], 1))
    templates = torch.cat([templates, temp_vis], dim=-1)
    assert templates.shape[-1] == 3
    assert gt_keypoints.shape[-1] == 3

    rows = gt_keypoints.size(0)
    cols = templates.size(0)

    if rows * cols == 0:
        return gt_keypoints.new(rows, 1) if is_aligned else gt_keypoints.new(rows, cols)

    if True:
        # TODO(xiao): not efficient
        area2 = cal_area_2_torch(templates)
        overlaps = []
        for k in gt_keypoints:
            overlaps.append(
                oks_iou_tensor(k.view(-1), templates.view(templates.shape[0], -1), cal_area_2(k), area2))

        ious = torch.stack(overlaps, dim=0)

        # area1 = []
        # area1.append(cal_area_2(k))
        # area1 = torch.stack(area1, dim=-1)

    else:
        lt = torch.max(gt_keypoints[:, None, :2], templates[:, :2])  # [rows, cols, 2]
        rb = torch.min(gt_keypoints[:, None, 2:], templates[:, 2:])  # [rows, cols, 2]

        wh = (rb - lt + 1).clamp(min=0)  # [rows, cols, 2]
        overlap = wh[:, :, 0] * wh[:, :, 1]
        area1 = (gt_keypoints[:, 2] - gt_keypoints[:, 0] + 1) * (
                gt_keypoints[:, 3] - gt_keypoints[:, 1] + 1)

        if mode == 'iou':
            area2 = (templates[:, 2] - templates[:, 0] + 1) * (
                    templates[:, 3] - templates[:, 1] + 1)
            ious = overlap / (area1[:, None] + area2 - overlap)
        else:
            ious = overlap / (area1[:, None])

    return ious


class MaxOksIoUAssigner(BaseAssigner):
    """Assign a corresponding gt bbox or background to each bbox.

    Each proposals will be assigned with `-1`, `0`, or a positive integer
    indicating the ground truth index.

    - -1: don't care
    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        pos_iou_thr (float): IoU threshold for positive bboxes.
        neg_iou_thr (float or tuple): IoU threshold for negative bboxes.
        min_pos_iou (float): Minimum iou for a bbox to be considered as a
            positive bbox. Positive samples can have smaller IoU than
            pos_iou_thr due to the 4th step (assign max IoU sample to each gt).
        gt_max_assign_all (bool): Whether to assign all bboxes with the same
            highest overlap with some gt to that gt.
        ignore_iof_thr (float): IoF threshold for ignoring bboxes (if
            `gt_bboxes_ignore` is specified). Negative values mean not
            ignoring any bboxes.
        ignore_wrt_candidates (bool): Whether to compute the iof between
            `bboxes` and `gt_bboxes_ignore`, or the contrary.
    """

    def __init__(self,
                 pos_iou_thr,
                 neg_iou_thr,
                 min_pos_iou=.0,
                 gt_max_assign_all=True,
                 ignore_iof_thr=-1,
                 ignore_wrt_candidates=True,
                 assign_fg_per_anchor=False,
                 assign_fg_per_level=False,
                 ):
        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr
        self.min_pos_iou = min_pos_iou
        self.gt_max_assign_all = gt_max_assign_all
        self.ignore_iof_thr = ignore_iof_thr
        self.ignore_wrt_candidates = ignore_wrt_candidates
        self.assign_fg_per_anchor = assign_fg_per_anchor
        self.assign_fg_per_level = assign_fg_per_level

    def assign(self, templates, templates_scales, anchor_infos, gt_keypoints, gt_labels=None):
        """Assign gt to bboxes.

        This method assign a gt bbox to every bbox (proposal/anchor), each bbox
        will be assigned with -1, 0, or a positive number. -1 means don't care,
        0 means negative sample, positive number is the index (1-based) of
        assigned gt.
        The assignment is done in following steps, the order matters.

        1. assign every bbox to -1
        2. assign proposals whose iou with all gts < neg_iou_thr to 0
        3. for each bbox, if the iou with its nearest gt >= pos_iou_thr,
           assign it to that bbox
        4. for each gt bbox, assign its nearest proposals (may be more than
           one) to itself

        Args:
            bboxes (Tensor): Bounding boxes to be assigned, shape(n, 4).
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 4).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.
        """

        # assert gt_bboxes.shape[0] == gt_keypoints.shape[0]

        if templates.shape[0] == 0 or gt_keypoints.shape[0] == 0:
            raise ValueError('No gt or bboxes')
        # bboxes = bboxes[:, :4]
        overlaps = kpt_overlaps(gt_keypoints, templates)

        # if (self.ignore_iof_thr > 0) and (gt_bboxes_ignore is not None) and (
        #         gt_bboxes_ignore.numel() > 0):
        #     assert 0
        #     # TODO(xiao): not implemented
        #     if self.ignore_wrt_candidates:
        #         ignore_overlaps = kpt_overlaps(
        #             templates, gt_bboxes_ignore, mode='iof')
        #         ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
        #     else:
        #         ignore_overlaps = kpt_overlaps(
        #             gt_bboxes_ignore, templates, mode='iof')
        #         ignore_max_overlaps, _ = ignore_overlaps.max(dim=0)
        #     overlaps[:, ignore_max_overlaps > self.ignore_iof_thr] = -1

        assign_result, num_fg_assigned, num_gts = self.assign_wrt_overlaps(overlaps, anchor_infos, gt_labels)
        #return assign_result, num_fg_assigned, num_gts
        return assign_result

    def assign_wrt_overlaps(self, overlaps, anchor_infos, gt_labels=None):
        """Assign w.r.t. the overlaps of bboxes with gts.

        Args:
            overlaps (Tensor): Overlaps between k gt_bboxes and n bboxes,
                shape(k, n).
            gt_labels (Tensor, optional): Labels of k gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        if overlaps.numel() == 0:
            raise ValueError('No gt or proposals')

        num_gts, num_templates = overlaps.size(0), overlaps.size(1)

        # 1. assign -1 by default
        assigned_gt_inds = overlaps.new_full((num_templates, ),
                                             -1,
                                             dtype=torch.long)

        # for each anchor, which gt best overlaps with it
        # for each anchor, the max iou of all gts
        max_overlaps, argmax_overlaps = overlaps.max(dim=0)
        # for each gt, which anchor best overlaps with it
        # for each gt, the max iou of all proposals


        # for each GT, find the top-9 anchors from all anchors
        top_num = 9
        gt_max_overlaps_N = []
        gt_argmax_overlaps_N = []
        overlaps_ = overlaps.clone()
        for n in range(top_num):
            gt_max_overlaps_, gt_argmax_overlaps_ = overlaps_.max(dim=1)
            gt_max_overlaps_N.append(gt_max_overlaps_)
            gt_argmax_overlaps_N.append(gt_argmax_overlaps_)
            for i in range(num_gts):
                overlaps_[i, gt_argmax_overlaps_[i]] = 0

        if self.assign_fg_per_anchor:
            featmap_sizes = anchor_infos['featmap_sizes']
            point_anchor_num = anchor_infos['point_anchor_num']
            gt_max_overlaps_anchor_list = []
            gt_argmax_overlaps_anchor_list = []
            if self.assign_fg_per_level:
                lvl_start = 0
                num_level = len(featmap_sizes)
                for n_l in range(num_level):
                    level_size = featmap_sizes[n_l][0] * featmap_sizes[n_l][1] * point_anchor_num
                    lvl_overlaps = overlaps[:, lvl_start:lvl_start + level_size]
                    for n_a in range(point_anchor_num):
                        a_overlaps = lvl_overlaps[:, n_a::point_anchor_num]
                        gt_max_overlaps_anchor, gt_argmax_overlaps_anchor_ = a_overlaps.max(dim=1)
                        gt_argmax_overlaps_anchor = gt_argmax_overlaps_anchor_ * point_anchor_num + n_a + lvl_start
                        gt_max_overlaps_anchor_list.append(gt_max_overlaps_anchor)
                        gt_argmax_overlaps_anchor_list.append(gt_argmax_overlaps_anchor)
                    lvl_start = lvl_start + level_size
            else:
                # for each GT, find the highest IoU anchor for each class of the anchors.
                for n_a in range(point_anchor_num):
                    a_overlaps = overlaps[:, n_a::point_anchor_num]
                    gt_max_overlaps_anchor, gt_argmax_overlaps_anchor_ = a_overlaps.max(dim=1)
                    gt_argmax_overlaps_anchor = gt_argmax_overlaps_anchor_ * point_anchor_num + n_a
                    gt_max_overlaps_anchor_list.append(gt_max_overlaps_anchor)
                    gt_argmax_overlaps_anchor_list.append(gt_argmax_overlaps_anchor)

        # 2. assign negative: below
        if isinstance(self.neg_iou_thr, float):
            assigned_gt_inds[(max_overlaps >= 0)
                             & (max_overlaps < self.neg_iou_thr)] = 0
        else:
            assert 0

        # 3. assign positive: above positive IoU threshold
        pos_inds = max_overlaps >= self.pos_iou_thr
        assigned_gt_inds[pos_inds] = argmax_overlaps[pos_inds] + 1

        # 4. assign fg: for each gt, proposals with highest IoU
        num_fg_assigned = overlaps.new_zeros((1))
        for i in range(num_gts):
            for n_t in range(top_num):
                gt_max_overlaps = gt_max_overlaps_N[n_t]
                gt_argmax_overlaps = gt_argmax_overlaps_N[n_t]
                if gt_max_overlaps[i] >= self.min_pos_iou:
                    if gt_max_overlaps[i] < self.pos_iou_thr:
                        num_fg_assigned = num_fg_assigned + 1.0
                    if self.gt_max_assign_all:
                        max_iou_inds = overlaps[i, :] == gt_max_overlaps[i]
                        # debug_b = torch.sum(max_iou_inds)
                        assigned_gt_inds[max_iou_inds] = i + 1
                    else:
                        assigned_gt_inds[gt_argmax_overlaps[i]] = i + 1
            if self.assign_fg_per_anchor:
                for sss in range(len(gt_argmax_overlaps_anchor_list)):
                    max_iou_inds_ = gt_argmax_overlaps_anchor_list[sss][i]
                    assigned_gt_inds[max_iou_inds_] = i + 1
        num_fg_assigned = num_fg_assigned / top_num

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_zeros((num_templates, ))
            # debug_c = assigned_gt_inds > 0
            # debug_d = torch.nonzero(debug_c)
            # debug_e = debug_d.squeeze()
            pos_inds = torch.nonzero(assigned_gt_inds > 0).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None

        return AssignResult(
            num_gts, assigned_gt_inds, max_overlaps, labels=assigned_labels), \
               num_fg_assigned, num_fg_assigned.new_zeros((1)) + num_gts
