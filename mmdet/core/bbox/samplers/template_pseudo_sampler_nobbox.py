import torch

from .base_sampler import BaseSampler
from .template_sampling_result import TemplateSamplingResult


class TemplatePseudoSamplerNobbox(BaseSampler):

    def __init__(self, **kwargs):
        pass

    def _sample_pos(self, **kwargs):
        raise NotImplementedError

    def _sample_neg(self, **kwargs):
        raise NotImplementedError

    def sample(self, assign_result, templates, templates_scales, gt_keypoints, **kwargs):
        pos_inds = torch.nonzero(
            assign_result.gt_inds > 0).squeeze(-1).unique()
        neg_inds = torch.nonzero(
            assign_result.gt_inds == 0).squeeze(-1).unique()
        # TODO(xiao): gt_flags seems useless?
        gt_flags = templates.new_zeros(templates.shape[0], dtype=torch.uint8)
        sampling_result = TemplateSamplingResult(pos_inds, neg_inds, templates, templates_scales, gt_keypoints,
                                                 assign_result, gt_flags)
        return sampling_result
