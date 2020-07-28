import torch


class TemplateSamplingResult(object):

    def __init__(self, pos_inds, neg_inds, templates, templates_scales, gt_keypoints, assign_result,
                 gt_flags):
        self.pos_inds = pos_inds
        self.neg_inds = neg_inds
        self.pos_templates = templates[pos_inds]
        self.neg_templates = templates[neg_inds]

        self.pos_templates_scales = templates_scales[pos_inds]
        self.neg_templates_scales = templates_scales[neg_inds]

        # TODO(xiao): useless?
        # self.pos_is_gt = gt_flags[pos_inds]

        self.num_gts = gt_keypoints.shape[0]
        self.pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1
        self.pos_gt_keypoints = gt_keypoints[self.pos_assigned_gt_inds, :]
        if assign_result.labels is not None:
            self.pos_gt_labels = assign_result.labels[pos_inds]
        else:
            self.pos_gt_labels = None

    # @property
    # TODO(xiao): useless?
    # def bboxes(self):
    #     return torch.cat([self.pos_bboxes, self.neg_bboxes])

class TemplateSamplingResultWithBbx(object):
    
    def __init__(self, pos_inds, neg_inds, templates, templates_bbx, templates_scales, gt_keypoints, 
                 gt_bboxes, assign_result, gt_flags):
        self.pos_inds = pos_inds
        self.neg_inds = neg_inds
        self.pos_templates = templates[pos_inds]
        self.neg_templates = templates[neg_inds]
        self.pos_templates_bbx = templates_bbx[pos_inds]
        self.neg_templates_bbx = templates_bbx[neg_inds]

        self.pos_templates_scales = templates_scales[pos_inds]
        self.neg_templates_scales = templates_scales[neg_inds]

        # TODO(xiao): useless?
        # self.pos_is_gt = gt_flags[pos_inds]

        self.num_gts = gt_keypoints.shape[0]
        self.num_gts_bbx = gt_bboxes.shape[0]
        self.pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1
        self.pos_gt_keypoints = gt_keypoints[self.pos_assigned_gt_inds, :]
        self.pos_gt_bboxes = gt_bboxes[self.pos_assigned_gt_inds, :]
        if assign_result.labels is not None:
            self.pos_gt_labels = assign_result.labels[pos_inds]
        else:
            self.pos_gt_labels = None