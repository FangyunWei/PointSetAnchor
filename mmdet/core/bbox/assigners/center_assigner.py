import torch

from .assign_result import AssignResult
from .base_assigner import BaseAssigner


class CenterAreaAssigner(object):

    def __init__(self, 
                 pos_area_ratio=0.4,
                 neg_area_ratio=0.5,
                 strides=(8, 16, 32),
                 scale_range_list=((0, 64), (64, 128), (128, 256))):
        super(CenterAreaAssigner, self).__init__()

        self.pos_area_ratio = pos_area_ratio
        self.neg_area_ratio = neg_area_ratio
        self.strides = strides
        self.scale_range_list = scale_range_list

    def assign(self,
               anchors,
               gt_bboxes,
               featmap_size_list,
               inside_flag,
               gt_bboxes_ignore=None,
               gt_labels=None):

        num_total_samples = anchors.size(0)
        device = anchors.device
        total_pixels = sum([fs[0] * fs[1] for fs in featmap_size_list])
        assert num_total_samples % total_pixels == 0
        samples_per_pixel = num_total_samples // total_pixels

        num_total_samples = len(inside_flag.nonzero().squeeze())

        # TODO: check the type of gt_bboxes_ignore
        assert gt_bboxes_ignore is None
        if gt_bboxes_ignore is not None:
            gt_bboxes = gt_bboxes[~gt_bboxes_ignore]
            if gt_labels is not None:
                gt_labels = gt_labels[~gt_bboxes_ignore]

        num_gts = gt_bboxes.shape[0]
        if num_gts == 0:
            assigned_gt_inds = torch.zeros((num_total_samples, ), 
                                            dtype=torch.long,
                                            device=device)
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = torch.zeros((num_total_samples, ), 
                                               dtype=torch.long,
                                               device=device)
            return AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_labels)

        assigned_gt_inds_list = []

        gt_inds = torch.arange(num_gts, dtype=torch.long, device=device)
        gt_areas = torch.sqrt((gt_bboxes[:, 2] - gt_bboxes[:, 0]) * 
                              (gt_bboxes[:, 3] - gt_bboxes[:, 1]))

        for (lower_bound, upper_bound), stride, featmap_size in zip(
                self.scale_range_list, self.strides, featmap_size_list):
            assigned_gt_inds = gt_inds.new_zeros(featmap_size)
            hit_ind = ((gt_areas >= lower_bound) &
                       (gt_areas <= upper_bound)).nonzero().flatten()
            if len(hit_ind) == 0:
                assigned_gt_inds_list.append(assigned_gt_inds.reshape(-1))
                continue

            _, hit_ind_order = torch.sort(-gt_areas[hit_ind])
            hit_ind = hit_ind[hit_ind_order]
            gt_bboxes_hit = gt_bboxes[hit_ind] / stride
            gt_inds_hit = gt_inds[hit_ind]
            half_w = (gt_bboxes_hit[:, 2] - gt_bboxes_hit[:, 0]) / 2
            half_h = (gt_bboxes_hit[:, 3] - gt_bboxes_hit[:, 1]) / 2

            pos_left = torch.ceil(gt_bboxes_hit[:, 0] + 
                (1 - self.pos_area_ratio) * half_w - 0.5).long(). \
                    clamp(0, featmap_size[1] - 1)
            pos_right = torch.floor(gt_bboxes_hit[:, 0] + 
                (1 + self.pos_area_ratio) * half_w - 0.5).long(). \
                    clamp(0, featmap_size[1] - 1)
            pos_top = torch.ceil(gt_bboxes_hit[:, 1] + 
                (1 - self.pos_area_ratio) * half_h - 0.5).long(). \
                    clamp(0, featmap_size[0] - 1)
            pos_down = torch.floor(gt_bboxes_hit[:, 1] + 
                (1 + self.pos_area_ratio) * half_h - 0.5).long(). \
                    clamp(0, featmap_size[0] - 1)

            for px1, py1, px2, py2, gt_ind in zip(pos_left, pos_top, 
                    pos_right, pos_down, gt_inds_hit):
                assigned_gt_inds[py1: py2+1, px1: px2+1] = gt_ind + 1

            assigned_gt_inds_list.append(assigned_gt_inds.reshape(-1))

        assigned_gt_inds_single = torch.cat(assigned_gt_inds_list)
        assigned_gt_inds = assigned_gt_inds_single.reshape(-1, 1). \
            repeat((1, samples_per_pixel)).reshape(-1)[inside_flag]

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_zeros((num_total_samples, ))
            pos_inds = torch.nonzero(assigned_gt_inds > 0).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None

        return AssignResult(
            num_gts, assigned_gt_inds, None, labels=assigned_labels)
