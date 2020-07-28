import torch

from ..bbox import TemplatePseudoSamplerNobbox, build_assigner
from ..utils import multi_apply


def template2delta(proposals, scales, gt, means, stds, use_out_scale):
    assert proposals.shape[0] == gt.shape[0]

    proposals = proposals.view(proposals.shape[0], -1, 2)
    proposals = proposals.float()
    gt = gt.float()

    px = proposals[:, :, 0]
    py = proposals[:, :, 1]

    gx = gt[:, 0::3]
    gy = gt[:, 1::3]
    gv = gt[:, 2::3]

    if use_out_scale:
        dx = (gx - px) / scales
        dy = (gy - py) / scales
    else:
        pw = torch.max(proposals[:, :, 0], -1)[0] - torch.min(proposals[:, :, 0], -1)[0] + 1.0
        ph = torch.max(proposals[:, :, 1], -1)[0] - torch.min(proposals[:, :, 1], -1)[0] + 1.0
        dx = (gx - px) / pw.view(-1, 1)
        dy = (gy - py) / ph.view(-1, 1)

    inds = gv == 0
    dx[inds] = 0
    dy[inds] = 0

    deltas = torch.stack([dx, dy], dim=-1)
    deltas = deltas.view(deltas.shape[0], -1)

    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)
    deltas = deltas.sub_(means).div_(stds)

    return deltas


def template_target_nobbox(anchor_list,
                    anchor_scale_list,
                    valid_flag_list,
                    gt_keypoints_list,
                    img_metas,
                    target_means,
                    target_stds,
                    anchor_infos,
                    use_out_scale,
                    cfg,
                    gt_labels_list=None,
                    label_channels=1,
                    sampling=True,
                    unmap_outputs=True):
    num_imgs = len(img_metas)
    assert len(anchor_list) == len(valid_flag_list) == num_imgs

    # anchor number of multi levels
    num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
    # concat all level anchors and flags to a single tensor
    for i in range(num_imgs):
        assert len(anchor_list[i]) == len(valid_flag_list[i])
        anchor_list[i] = torch.cat(anchor_list[i])
        valid_flag_list[i] = torch.cat(valid_flag_list[i])
        anchor_scale_list[i] = torch.cat(anchor_scale_list[i])

    if gt_labels_list is None:
        gt_labels_list = [None for _ in range(num_imgs)]
    (all_labels, all_label_weights, all_reg_targets, all_reg_weights,
     pos_inds_list, neg_inds_list) = multi_apply(
         template_target_single,
         anchor_list,
         anchor_scale_list,
         valid_flag_list,
         gt_keypoints_list,
         gt_labels_list,
         img_metas,
         target_means=target_means,
         target_stds=target_stds,
         anchor_infos=anchor_infos,
         use_out_scale=use_out_scale,
         cfg=cfg,
         sampling=sampling,
         unmap_outputs=unmap_outputs)
    # no valid anchors
    if any([labels is None for labels in all_labels]):
        return None
    # sampled anchors of all images
    num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
    num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])


    # split targets to a list w.r.t. multiple levels
    labels_list = images_to_levels(all_labels, num_level_anchors)
    label_weights_list = images_to_levels(all_label_weights, num_level_anchors)
    reg_targets_list = images_to_levels(all_reg_targets, num_level_anchors)
    reg_weights_list = images_to_levels(all_reg_weights, num_level_anchors)

    num_pos_list = []
    num_neg_list = []
    for l, w in zip(labels_list, label_weights_list):
        num_pos = torch.sum(l > 0)
        num_all = torch.sum(w > 0)
        num_neg = num_all - num_pos
        num_pos_list.append(num_pos.float())
        num_neg_list.append(num_neg.float())

    return (labels_list, label_weights_list, reg_targets_list,
            reg_weights_list, num_total_pos, num_total_neg, num_pos_list, num_neg_list)


def images_to_levels(target, num_level_anchors):
    target = torch.stack(target, 0)
    level_targets = []
    start = 0
    for n in num_level_anchors:
        end = start + n
        level_targets.append(target[:, start:end].squeeze(0))
        start = end
    return level_targets


def template_target_single(flat_anchors,
                           flat_anchors_scales,
                           valid_flags,
                           gt_keypoints,
                           gt_labels,
                           img_meta,
                           target_means,
                           target_stds,
                           anchor_infos,
                           use_out_scale,
                           cfg,
                           sampling=True,
                           unmap_outputs=True):
    inside_flags = template_inside_flags(flat_anchors, valid_flags,
                                         img_meta['img_shape'][:2],
                                         cfg.allowed_border)
    if not inside_flags.any():
        return (None, ) * 6
    # assign gt and sample anchors
    anchors = flat_anchors[inside_flags, :]
    anchors_scales = flat_anchors_scales[inside_flags, :]

    if sampling:
        assert 0
    else:
        template_assigner = build_assigner(cfg.assigner)
        assign_result = template_assigner.assign(anchors, anchors_scales, anchor_infos, gt_keypoints, gt_labels)
        template_sampler = TemplatePseudoSamplerNobbox()
        sampling_result = template_sampler.sample(assign_result, anchors, anchors_scales,
                                                  gt_keypoints.view(gt_keypoints.shape[0], -1))

    num_valid_anchors = anchors.shape[0]
    template_targets = torch.zeros_like(anchors)
    template_weights = torch.zeros_like(anchors)
    labels = anchors.new_zeros(num_valid_anchors, dtype=torch.long)
    label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

    pos_inds = sampling_result.pos_inds
    neg_inds = sampling_result.neg_inds
    if len(pos_inds) > 0:
        pos_template_targets = template2delta(sampling_result.pos_templates,
                                              sampling_result.pos_templates_scales,
                                              sampling_result.pos_gt_keypoints,
                                              target_means, target_stds, use_out_scale)
        template_targets[pos_inds, :] = pos_template_targets
        t_v = sampling_result.pos_gt_keypoints[:, 2::3].clone()
        t_v[t_v > 0.5] = 1.0
        t_v[t_v < 0.5] = 0.0
        t_v = torch.stack([t_v, t_v], dim=-1).reshape(t_v.shape[0], -1)
        template_weights[pos_inds, :] = t_v
        if gt_labels is None:
            assert 0
        else:
            labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        if cfg.pos_weight <= 0:
            label_weights[pos_inds] = 1.0
        else:
            label_weights[pos_inds] = cfg.pos_weight
    if len(neg_inds) > 0:
        label_weights[neg_inds] = 1.0

    # map up to original set of anchors
    if unmap_outputs:
        num_total_anchors = flat_anchors.size(0)
        labels = unmap(labels, num_total_anchors, inside_flags)
        label_weights = unmap(label_weights, num_total_anchors, inside_flags)
        template_targets = unmap(template_targets, num_total_anchors, inside_flags)
        template_weights = unmap(template_weights, num_total_anchors, inside_flags)

    return (labels, label_weights, template_targets, template_weights, pos_inds,
            neg_inds)


def template_inside_flags(flat_anchors, valid_flags, img_shape,
                          allowed_border=0):
    img_h, img_w = img_shape[:2]
    if allowed_border >= 0:
        inside_flags = valid_flags & \
            (flat_anchors[:, 0] >= -allowed_border).type(torch.uint8) & \
            (flat_anchors[:, 1] >= -allowed_border).type(torch.uint8) & \
            (flat_anchors[:, 2] < img_w + allowed_border).type(torch.uint8) & \
            (flat_anchors[:, 3] < img_h + allowed_border).type(torch.uint8)
    else:
        inside_flags = valid_flags
    return inside_flags


def unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if data.dim() == 1:
        ret = data.new_full((count, ), fill)
        ret[inds] = data
    else:
        new_size = (count, ) + data.size()[1:]
        ret = data.new_full(new_size, fill)
        ret[inds, :] = data
    return ret
