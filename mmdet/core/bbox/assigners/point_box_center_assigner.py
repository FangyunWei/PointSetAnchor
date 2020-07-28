import torch

from .base_assigner import BaseAssigner
from .assign_result import AssignResult
from mmcv_custom.vis import get_kpt_center
import numpy as np


class PointBoxCenterAssigner(BaseAssigner):
    """Assign a corresponding gt bbox or background to each point.

    Each proposals will be assigned with `0`, or a positive integer
    indicating the ground truth index.

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    """

    def __init__(self, scale=4, pos_num=3, pos_num_with_stride=False):
        self.scale = scale
        self.pos_num = pos_num
        self.pos_num_with_stride = pos_num_with_stride

    def assign(self, points, gt_keypoints, gt_keypoints_vis, center_type='shoulder', assign_type='stride',
               gt_keypoints_ignore=None, gt_labels=None):
        """Assign gt to bboxes.

        This method assign a gt bbox to every point, each bbox
        will be assigned with  0, or a positive number.
        0 means negative sample, positive number is the index (1-based) of
        assigned gt.
        The assignment is done in following steps, the order matters.

        1. assign every points to 0
        2. for each gt box, we find the k most closest points to the
            box center and assign the gt bbox to those points, we also record
            the minimum distance from each point to the closest gt box. When we
            assign the bbox to the points, we check whether its distance to the
            points is closest.

        Args:
            points (Tensor): points to be assigned, shape(n, 3) while last
                dimension stands for (x, y, stride).
            gt_keypoints (Tensor): Groundtruth boxes, shape (k, 4).
            gt_keypoints_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
            gt_labels (Tensor, optional): Label of gt_keypoints, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        if points.shape[0] == 0 or gt_keypoints.shape[0] == 0:
            raise ValueError('No gt or bboxes')
        points_range = torch.arange(points.shape[0])
        points_xy = points[:, :2]
        points_stride = points[:, 2]
        points_lvl = torch.log2(points_stride).int()  # [3...,4...,5...,6...,7...]
        lvl_min, lvl_max = points_lvl.min(), points_lvl.max()
        num_gts, num_points = gt_keypoints.shape[0], points.shape[0]

        # assign gt box
        gt_bboxes_x = 0.5 * (gt_keypoints[:, 0] + gt_keypoints[:, 2])
        gt_bboxes_y = 0.5 * (gt_keypoints[:, 1] + gt_keypoints[:, 3])
        if isinstance(center_type, str):
            center_index = get_kpt_center(center_type)
        else:
            center_index = center_type
        gt_poses_x = torch.mean(gt_keypoints[:, 4::2][:, center_index], dim=-1)
        gt_poses_y = torch.mean(gt_keypoints[:, 5::2][:, center_index], dim=-1)
        gt_poses_vis = torch.mean(gt_keypoints_vis[:, center_index], dim=-1)
        gt_keypoints_xy = torch.stack([gt_poses_x, gt_poses_y], -1)
        gt_bboxes_w = gt_keypoints[:, 2] - gt_keypoints[:, 0]
        gt_bboxes_h = gt_keypoints[:, 3] - gt_keypoints[:, 1]
        gt_bboxes_wh = torch.stack([gt_bboxes_w, gt_bboxes_h], -1)
        gt_bboxes_wh = torch.clamp(gt_bboxes_wh, min=1e-6)
        # TODO(xiao): should we -1 here? not that severe here since we only use it to compute the level.
        # TODO(yifan): use log 4 for 3 fpn output
        if assign_type == 'stride':
            gt_bboxes_lvl = (0.5 * (torch.log2(gt_bboxes_w / self.scale) + torch.log2(gt_bboxes_h / self.scale))).int()
        elif assign_type == 'diagonal':
            diagonal = lambda x: torch.sqrt(torch.pow(x[:, 3] - x[:, 1], 2) + torch.pow(x[:, 2] - x[:, 0], 2))
            gt_bboxes_lvl = torch.log2(diagonal(gt_keypoints) / (np.sqrt(2) * self.scale)).int()
        gt_bboxes_lvl_for_assign = torch.clamp(gt_bboxes_lvl, min=lvl_min, max=lvl_max + 4) - lvl_min
        gt_bboxes_lvl = torch.clamp(gt_bboxes_lvl, min=lvl_min, max=lvl_max)

        # stores the assigned gt index of each point
        assigned_gt_inds = points.new_zeros((num_points,), dtype=torch.long)
        # ignore neg idx in no kpts bbox in all level
        assigned_neg_gt_inds = points.new_zeros((num_points,), dtype=torch.long)
        # assigned_near_center_gt_inds = points.new_zeros((num_points,), dtype=torch.long)
        # ignore neg idx from other level
        assigned_ignore_lvl_gt_inds = points.new_zeros((num_points,), dtype=torch.long)
        # stores the assigned gt dist (to this point) of each point
        assigned_gt_dist = points.new_full((num_points,), float('inf'))

        for idx in range(num_gts):
            gt_lvl = gt_bboxes_lvl[idx]
            # get the index of points in this level
            lvl_idx = gt_lvl == points_lvl
            points_index = points_range[lvl_idx]

            if gt_poses_vis[idx] < 1:
                if torch.sum(gt_keypoints[idx][4:] > 0) == 0:
                    assigned_neg_gt_inds_mask = (points_xy[:, 0] > gt_keypoints[idx][0]).__and__(
                        points_xy[:, 0] < gt_keypoints[idx][2]).__and__(
                        points_xy[:, 1] < gt_keypoints[idx][3]).__and__(
                        points_xy[:, 1] > gt_keypoints[idx][1])
                    assigned_neg_gt_inds[assigned_neg_gt_inds_mask] = 1
                continue

            # get the points in this level
            lvl_points = points_xy[lvl_idx, :]
            # get the center point of gt
            gt_point = gt_keypoints_xy[[idx], :]
            # get width and height of gt
            gt_wh = gt_bboxes_wh[[idx], :]
            # compute the distance between gt center and all points in this level
            points_gt_dist = ((lvl_points-gt_point)/gt_wh).norm(dim=1)
            # find the nearest k points to gt center in this level
            if self.pos_num_with_stride and lvl_min == lvl_max:
                posnum = self.pos_num * int(gt_bboxes_lvl_for_assign[idx] + 1)
            else:
                posnum = self.pos_num
            min_dist, min_dist_index = torch.topk(-points_gt_dist, posnum)
            # the index of nearest k points to gt center in this level
            min_dist_points_index = points_index[min_dist_index]
            less_than_recorded_index = min_dist < assigned_gt_dist[min_dist_points_index]
            min_dist_points_index = min_dist_points_index[less_than_recorded_index]
            # assign the result
            assigned_gt_inds[min_dist_points_index] = idx+1
            assigned_gt_dist[min_dist_points_index] = min_dist[less_than_recorded_index]
            if min_dist_points_index.numel() > 0:
                for _i in min_dist_points_index:
                    assigned_ignore_lvl_gt_inds_mask = (points_xy[:, 0] == points_xy[_i, 0]).__and__(
                                points_xy[:, 1] == points_xy[_i, 1]).__and__(~lvl_idx)
                    assigned_ignore_lvl_gt_inds[assigned_ignore_lvl_gt_inds_mask] = 1

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_zeros((num_points,))
            pos_inds = torch.nonzero(assigned_gt_inds > 0).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None
        return AssignResult(num_gts, assigned_gt_inds, None,
                            labels=assigned_labels), assigned_neg_gt_inds, assigned_ignore_lvl_gt_inds
