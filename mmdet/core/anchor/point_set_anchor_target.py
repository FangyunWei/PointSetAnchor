import torch
import numpy as np

from ..bbox import (PointSetAnchorPseudoSampler, assign_and_sample, bbox2delta, points2delta,
                    pointdist2distdelta, build_assigner)
from ..utils import multi_apply
from mmcv import Config


def point_set_anchor_target(anchor_points_list,
                            anchor_points_count_list,
                            anchor_list,
                            valid_flag_list,
                            gt_bboxes_list,
                            img_metas,
                            target_means,
                            target_stds,
                            corner_number,
                            cfg,
                            featmap_sizes=None,
                            gt_bboxes_ignore_list=None,
                            gt_labels_list=None,
                            gt_masks_list=None,
                            label_channels=1,
                            sampling=True,
                            unmap_outputs=True):
    """Compute regression and classification targets for anchors.

    Args:
        anchor_list (list[list]): Multi level anchors of each image.
        valid_flag_list (list[list]): Multi level valid flags of each image.
        gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
        img_metas (list[dict]): Meta info of each image.
        target_means (Iterable): Mean value of regression targets.
        target_stds (Iterable): Std value of regression targets.
        cfg (dict): RPN train configs.

    Returns:
        tuple
    """
    num_imgs = len(img_metas)
    assert len(anchor_points_list) == len(anchor_list) == \
           len(anchor_points_count_list) == len(valid_flag_list) == num_imgs

    # anchor number of multi levels
    num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
    # concat all level anchors and flags to a single tensor
    for i in range(num_imgs):
        assert len(anchor_list[i]) == len(valid_flag_list[i]) == \
            len(anchor_points_list[i]) == len(anchor_points_count_list[i])
        anchor_list[i] = torch.cat(anchor_list[i])
        valid_flag_list[i] = torch.cat(valid_flag_list[i])
        anchor_points_list[i] = torch.cat(anchor_points_list[i])
        anchor_points_count_list[i] = torch.cat(anchor_points_count_list[i])

    # compute targets for each image
    if gt_bboxes_ignore_list is None:
        gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
    if gt_labels_list is None:
        gt_labels_list = [None for _ in range(num_imgs)]
    if gt_masks_list is None:
        gt_masks_list = [None for _ in range(num_imgs)]
    (all_labels, all_label_weights, all_bbox_targets, all_bbox_weights,
     all_point_dist_targets, all_point_dist_weights, all_point_dists_binary_targets,
     all_corner_targets, all_corner_weights, all_contour_targets, pos_inds_list,
     neg_inds_list) = multi_apply(
         anchor_target_single,
         anchor_points_list,
         anchor_points_count_list,
         anchor_list,
         valid_flag_list,
         gt_bboxes_list,
         gt_bboxes_ignore_list,
         gt_labels_list,
         gt_masks_list,
         img_metas,
         featmap_sizes=featmap_sizes,
         target_means=target_means,
         target_stds=target_stds,
         corner_number=corner_number,
         cfg=cfg,
         label_channels=label_channels,
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
    bbox_targets_list = images_to_levels(all_bbox_targets, num_level_anchors)
    bbox_weights_list = images_to_levels(all_bbox_weights, num_level_anchors)
    point_dist_targets_list = images_to_levels(all_point_dist_targets, num_level_anchors)
    point_dist_weights_list = images_to_levels(all_point_dist_weights, num_level_anchors)
    point_dists_binary_targets_list = images_to_levels(all_point_dists_binary_targets, num_level_anchors)
    corner_targets_list = images_to_levels(all_corner_targets, num_level_anchors)
    corner_weights_list = images_to_levels(all_corner_weights, num_level_anchors)
    contour_targets_list = images_to_levels_by_list(all_contour_targets, num_level_anchors)
    anchor_list = images_to_levels(anchor_list, num_level_anchors)
    anchor_points_list = images_to_levels(anchor_points_list, num_level_anchors)

    return (labels_list, label_weights_list, bbox_targets_list,
            bbox_weights_list, point_dist_targets_list, point_dist_weights_list,
            point_dists_binary_targets_list, corner_targets_list, corner_weights_list,
            contour_targets_list, anchor_list, anchor_points_list,
            num_total_pos, num_total_neg)

def images_to_levels_by_list(target, num_level_anchors):
    """Convert targets by image to targets by feature level.

    [target_img0, target_img1] -> [target_level0, target_level1, ...]
    """
    # concat all lists in target
    level_targets = []
    start = 0
    for n in num_level_anchors:
        end = start + n
        current_target = []
        for t in target:
            current_target.extend(t[start:end])
        level_targets.append(current_target)
        start = end
    return level_targets


def images_to_levels(target, num_level_anchors):
    """Convert targets by image to targets by feature level.

    [target_img0, target_img1] -> [target_level0, target_level1, ...]
    """
    target = torch.stack(target, 0)
    level_targets = []
    start = 0
    for n in num_level_anchors:
        end = start + n
        tmp = target[:, start:end]
        tmp2 = target[:, start:end].squeeze(0)
        level_targets.append(target[:, start:end].squeeze(0))
        start = end
    return level_targets


def anchor_target_single(flat_points,
                         flat_points_count,
                         flat_anchors,
                         valid_flags,
                         gt_bboxes,
                         gt_bboxes_ignore,
                         gt_labels,
                         gt_masks,
                         img_meta,
                         featmap_sizes,
                         target_means,
                         target_stds,
                         corner_number,
                         cfg,   #cfg = train_cfg
                         label_channels=1,
                         sampling=True,
                         unmap_outputs=True):
    inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                       img_meta['img_shape'][:2],
                                       cfg.allowed_border)
    assert corner_number == 4
    if not inside_flags.any():
        return (None, ) * 6
    # assign gt and sample anchors
    anchors = flat_anchors[inside_flags, :]
    points = flat_points[inside_flags, :]
    points_count = flat_points_count[inside_flags, :]

    if sampling:
        assign_result, sampling_result = assign_and_sample(
            anchors, gt_bboxes, gt_bboxes_ignore, None, cfg)
    else:
        bbox_assigner = build_assigner(cfg.assigner)

        if cfg.assigner.type == 'PointSetAnchorCenterAssigner':
            assign_result = bbox_assigner.assign(anchors, gt_bboxes,
                                                gt_bboxes_ignore, gt_labels)
        else:
            raise NotImplementedError
        bbox_sampler = PointSetAnchorPseudoSampler()
        sampling_result = bbox_sampler.sample(assign_result, anchors, gt_bboxes,
                                              points, points_count, gt_masks)

    num_valid_anchors = anchors.shape[0]
    bbox_targets = torch.zeros_like(anchors)
    bbox_weights = torch.zeros_like(anchors)
    labels = anchors.new_zeros(num_valid_anchors, dtype=torch.long)
    label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)
    # assigned target points on contour
    point_dists_targets = points.new_zeros(points.shape[0], points.shape[1]//2)
    point_dists_binary_targets = points.new_zeros(points.shape[0], points.shape[1]//2)
    point_dists_weights = points.new_zeros(points.shape[0], points.shape[1]//2)
    # corner targets
    corner_targets = points.new_zeros(points.shape[0], corner_number * 2)
    corner_weights = points.new_zeros(points.shape[0], corner_number * 2)
    # raw target contour
    contour_targets = np.empty(num_valid_anchors, dtype=object)

    pos_inds = sampling_result.pos_inds
    neg_inds = sampling_result.neg_inds
    if len(pos_inds) > 0:
        pos_bbox_targets = bbox2delta(sampling_result.pos_bboxes,
                                      sampling_result.pos_gt_bboxes,
                                      target_means, target_stds)
        # assign anchor point to GT point
        assigned_corner_gts, assigned_point_dists, assigned_binary_mask =\
            corner_anchor_points_assign(sampling_result.pos_masks,
                                          sampling_result.pos_points_count,
                                          sampling_result.pos_gt_masks,
                                          corner_number,
                                          cfg.get("points_assigner", Config(dict(type='CornerPointWithLine'))))
        # encdoing targets
        # TODO: corner should be point to line, not point to point?
        pos_corner_points = get_corner_points_from_anchor_points(sampling_result.pos_masks)
        pos_corner_targets = points2delta(pos_corner_points,
                                          sampling_result.pos_bboxes,
                                          assigned_corner_gts,
                                          0, 1)
        pos_point_dists_targets = pointdist2distdelta(assigned_point_dists,
                                                      sampling_result.pos_bboxes,
                                                      sampling_result.pos_points_count,
                                                      0, 1)
        bbox_targets[pos_inds, :] = pos_bbox_targets
        bbox_weights[pos_inds, :] = 1.0
        point_dists_targets[pos_inds, :] = pos_point_dists_targets
        point_dists_binary_targets[pos_inds, :] = assigned_binary_mask
        point_dists_weights[pos_inds, :] = 1.0
        corner_targets[pos_inds, :] = pos_corner_targets
        corner_weights[pos_inds, :] = 1.0
        contour_targets[pos_inds.cpu().numpy()] = sampling_result.pos_gt_masks
        #for i in range(len(pos_inds)):
        #    idx = pos_inds[i]
        #    contour_targets[idx] = sampling_result.pos_gt_masks[i]
        if gt_labels is None:
            labels[pos_inds] = 1
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
        bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
        bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)
        point_dists_targets = unmap(point_dists_targets, num_total_anchors, inside_flags)
        point_dists_weights = unmap(point_dists_weights, num_total_anchors, inside_flags)
        point_dists_binary_targets = unmap(point_dists_binary_targets, num_total_anchors, inside_flags)
        corner_targets = unmap(corner_targets, num_total_anchors, inside_flags)
        corner_weights = unmap(corner_weights, num_total_anchors, inside_flags)
        contour_targets = unmap_by_list(contour_targets.tolist(), num_total_anchors, inside_flags)

    return (labels, label_weights, bbox_targets, bbox_weights, point_dists_targets,
            point_dists_weights, point_dists_binary_targets, corner_targets, corner_weights,
            contour_targets, pos_inds, neg_inds)


def get_corner_points_from_anchor_points(point_proposals):
    point_proposals_x = point_proposals[:, 0::2]
    point_proposals_y = point_proposals[:, 1::2]
    corner_pts_x1 = torch.min(point_proposals_x, dim=1).values.unsqueeze(1)
    corner_pts_x2 = torch.max(point_proposals_x, dim=1).values.unsqueeze(1)
    corner_pts_y1 = torch.min(point_proposals_y, dim=1).values.unsqueeze(1)
    corner_pts_y2 = torch.max(point_proposals_y, dim=1).values.unsqueeze(1)
    corner_pts_x1y1 = torch.cat((corner_pts_x1, corner_pts_y1), dim=1)
    corner_pts_x1y2 = torch.cat((corner_pts_x1, corner_pts_y2), dim=1)
    corner_pts_x2y1 = torch.cat((corner_pts_x2, corner_pts_y1), dim=1)
    corner_pts_x2y2 = torch.cat((corner_pts_x2, corner_pts_y2), dim=1)
    # [note] clockwise
    corner_pts = torch.cat((corner_pts_x1y1, corner_pts_x2y1,
                            corner_pts_x2y2, corner_pts_x1y2), dim=1)
    return corner_pts


def get_anchor_point_to_mask_dist_wside(starts, ends, points, mask_gts,
                                        slope_epsilon=1e-5):
    if len(points.shape) == 1:
        points = points.unsqueeze(0)
    sample_num = len(points)
    points_num = points.shape[1] // 2
    points_binary_mask = points.new_zeros(sample_num, points_num)
    mask_gts_x = mask_gts[0::2]
    mask_gts_y = mask_gts[1::2]
    mask_gts_x_shift = torch.roll(mask_gts_x, -1)
    mask_gts_y_shift = torch.roll(mask_gts_y, -1)
    slope = (mask_gts_y_shift - mask_gts_y) / \
             (mask_gts_x_shift - mask_gts_x + slope_epsilon)
    bias = mask_gts_y_shift - slope * mask_gts_x_shift

    points_x = points[:, 0::2].unsqueeze(-1)
    points_y = points[:, 1::2].unsqueeze(-1)
    points_x_filter = ((points_x >= mask_gts_x) & (points_x <= mask_gts_x_shift)) |\
                      ((points_x <= mask_gts_x) & (points_x >= mask_gts_x_shift))
    inter_y = slope * points_x + bias
    inter_y2point_y_dist = inter_y - points_y
    # set unsatisfied points dist to a big number
    inter_y2point_y_dist[~points_x_filter] = 1e8
    inter_y2point_y_absdist_min_idx = torch.argmin(torch.abs(inter_y2point_y_dist), dim=-1).unsqueeze(-1)
    points_match_res = torch.gather(inter_y2point_y_dist, 2, inter_y2point_y_absdist_min_idx).squeeze(-1)

    #TODO: using starts and ends to limit points regression???
    #TODO: setting unsatisified points in points_match_res to 0???
    points_binary_mask_filter = (points_x.squeeze(-1) > starts.unsqueeze(-1)) &\
                                (points_x.squeeze(-1) < ends.unsqueeze(-1))
    points_binary_mask[points_binary_mask_filter] = 1

    # set unmathced points to 0
    unmatched_idx = points_match_res == 1e8
    points_match_res[unmatched_idx] = 0
    points_binary_mask[unmatched_idx] = 0
    return points_match_res, points_binary_mask


def get_anchor_point_to_mask_dist_hside(starts, ends, points, mask_gts,
                                          slope_epsilon=1e-5):
    if len(points.shape) == 1:
        points = points.unsqueeze(0)
    sample_num = len(points)
    points_num = points.shape[1] // 2
    points_binary_mask = points.new_zeros(sample_num, points_num)
    mask_gts_x = mask_gts[0::2]
    mask_gts_y = mask_gts[1::2]
    mask_gts_x_shift = torch.roll(mask_gts_x, -1)
    mask_gts_y_shift = torch.roll(mask_gts_y, -1)
    slope = (mask_gts_y_shift - mask_gts_y) / \
            (mask_gts_x_shift - mask_gts_x + slope_epsilon)
    bias = mask_gts_y_shift - slope * mask_gts_x_shift

    points_x = points[:, 0::2].unsqueeze(-1)
    points_y = points[:, 1::2].unsqueeze(-1)

    points_y_filter = ((points_y >= mask_gts_y) & (points_y <= mask_gts_y_shift)) |\
                      ((points_y <= mask_gts_y) & (points_y >= mask_gts_y_shift))
    inter_x = (points_y - bias) / (slope + slope_epsilon)
    inter_x2point_x_dist = inter_x - points_x
    # set unsatisfied points dist to a big number
    inter_x2point_x_dist[~points_y_filter] = 1e8
    inter_x2point_x_absdist_min_idx = torch.argmin(torch.abs(inter_x2point_x_dist), dim=-1).unsqueeze(-1)
    points_match_res = torch.gather(inter_x2point_x_dist, 2, inter_x2point_x_absdist_min_idx).squeeze(-1)

    #TODO: using starts and ends to limit points regression???
    #TODO: setting unsatisified points in points_match_res to 0???
    points_binary_mask_filter = (points_y.squeeze(-1) > starts.unsqueeze(-1)) & \
                                (points_y.squeeze(-1) < ends.unsqueeze(-1))
    points_binary_mask[points_binary_mask_filter] = 1

    # set unmathced points to 0
    unmatched_idx = points_match_res == 1e8
    points_match_res[unmatched_idx] = 0
    points_binary_mask[unmatched_idx] = 0
    return points_match_res, points_binary_mask


def corner_anchor_points_assign_corner_with_line(point_proposals,
                                                   point_proposals_counter,
                                                   mask_gts,
                                                   corner_num,
                                                   points_assigner_cfg):
    assert corner_num == 4
    corner_match = points_assigner_cfg.get('corner_match', 'NearestPoint')
    device = point_proposals.device
    anchor_point_num = point_proposals.shape[1]//2
    assigned_mask_gts = point_proposals.new_zeros((point_proposals.shape[0],
                                                   anchor_point_num))
    assigned_binary_mask_gts = point_proposals.new_zeros((point_proposals.shape[0],
                                                          anchor_point_num))
    assigned_corner_gts = point_proposals.new_zeros((point_proposals.shape[0],
                                                     corner_num * 2))
    mask_gts_sum = torch.Tensor([torch.sum(row) for row in mask_gts])
    for label in torch.unique(mask_gts_sum):
        cur_idx = (mask_gts_sum == label).nonzero().squeeze(-1)
        cur_points = point_proposals[cur_idx, :]
        cur_points_counter = point_proposals_counter[cur_idx, :]
        cur_mask_points = mask_gts[int(cur_idx[0])].to(device)
        assert len(cur_mask_points) % 2 == 0
        cur_mask_points_x = cur_mask_points[0::2]
        cur_mask_points_y = cur_mask_points[1::2]
        # get corner point target
        cur_corner_points = get_corner_points_from_anchor_points(cur_points)
        if corner_match == 'NearestPoint':
            cur_corner_points_dist = torch.abs(cur_corner_points[:, 0::2, None] - cur_mask_points_x) + \
                                     torch.abs(cur_corner_points[:, 1::2, None] - cur_mask_points_y)
            cur_corner_points_dist_argmin = torch.argmin(cur_corner_points_dist, dim=-1)
            cur_corner_points_target_x = cur_mask_points_x[cur_corner_points_dist_argmin]
            cur_corner_points_target_y = cur_mask_points_y[cur_corner_points_dist_argmin]
        else:
            raise NotImplementedError
        assigned_corner_gts[cur_idx, 0::2] = cur_corner_points_target_x
        assigned_corner_gts[cur_idx, 1::2] = cur_corner_points_target_y

        # get anchor target
        cur_corner_target_lt_x = cur_corner_points_target_x[:, 0]
        cur_corner_target_lt_y = cur_corner_points_target_y[:, 0]
        cur_corner_target_rt_x = cur_corner_points_target_x[:, 1]
        cur_corner_target_rt_y = cur_corner_points_target_y[:, 1]
        cur_corner_target_rb_x = cur_corner_points_target_x[:, 2]
        cur_corner_target_rb_y = cur_corner_points_target_y[:, 2]
        cur_corner_target_lb_x = cur_corner_points_target_x[:, 3]
        cur_corner_target_lb_y = cur_corner_points_target_y[:, 3]

        cur_corner_points_x1 = cur_corner_points[:, 0]
        cur_corner_points_x2 = cur_corner_points[:, 2]

        cur_corner_points_w = cur_corner_points_x2 - cur_corner_points_x1
        # TODO:for each ratio, better implement?
        cur_points_assign_res = assigned_mask_gts.new_zeros((len(cur_idx), anchor_point_num))
        cur_points_binary_mask_res = assigned_mask_gts.new_zeros((len(cur_idx), anchor_point_num))
        cur_ws = cur_corner_points_w.unique()
        for w in cur_ws:
            cur_ratio_idx = (cur_corner_points_w == w).nonzero().squeeze(-1)
            cur_ratio_points = cur_points[cur_ratio_idx, :]
            cur_ratio_points_counter = cur_points_counter[cur_ratio_idx, :][0]
            # idx in each side
            cur_ratio_points_counter_idx = []
            len_sum = 0
            for i in range(4):
                len_sum += int(cur_ratio_points_counter[i])
                cur_ratio_points_counter_idx.append(len_sum)
            cur_ratio_top_target, cur_ratio_top_mask =\
                get_anchor_point_to_mask_dist_wside(cur_corner_target_lt_x[cur_ratio_idx],
                                                      cur_corner_target_rt_x[cur_ratio_idx],
                                                      cur_ratio_points[:, 0:cur_ratio_points_counter_idx[0]*2],
                                                      cur_mask_points)
            cur_ratio_right_target, cur_ratio_right_mask = \
                get_anchor_point_to_mask_dist_hside(cur_corner_target_rt_y[cur_ratio_idx],
                                                      cur_corner_target_rb_y[cur_ratio_idx],
                                                      cur_ratio_points[:,
                                                      cur_ratio_points_counter_idx[0]*2:cur_ratio_points_counter_idx[1]*2],
                                                      cur_mask_points)
            cur_ratio_bottom_target, cur_ratio_bottom_mask = \
                get_anchor_point_to_mask_dist_wside(cur_corner_target_lb_x[cur_ratio_idx],
                                                      cur_corner_target_rb_x[cur_ratio_idx],
                                                      cur_ratio_points[:,
                                                      cur_ratio_points_counter_idx[1]*2:cur_ratio_points_counter_idx[2]*2],
                                                      cur_mask_points)
            cur_ratio_left_target, cur_ratio_left_mask = \
                get_anchor_point_to_mask_dist_hside(cur_corner_target_lt_y[cur_ratio_idx],
                                                      cur_corner_target_lb_y[cur_ratio_idx],
                                                      cur_ratio_points[:,
                                                      cur_ratio_points_counter_idx[2]*2:cur_ratio_points_counter_idx[3]*2],
                                                      cur_mask_points)
            cur_ratio_target = torch.cat([cur_ratio_top_target, cur_ratio_right_target,
                                          cur_ratio_bottom_target, cur_ratio_left_target], dim=1)
            cur_points_assign_res[cur_ratio_idx, :] = cur_ratio_target
            cur_ratio_binary_mask = torch.cat([cur_ratio_top_mask, cur_ratio_right_mask,
                                               cur_ratio_bottom_mask, cur_ratio_left_mask], dim=1)
            cur_points_binary_mask_res[cur_ratio_idx, :] = cur_ratio_binary_mask

        assigned_mask_gts[cur_idx, :] = cur_points_assign_res
        assigned_binary_mask_gts[cur_idx, :] = cur_points_binary_mask_res

    return assigned_corner_gts, assigned_mask_gts, assigned_binary_mask_gts


def corner_anchor_points_assign(point_proposals, point_proposals_counter,
                                  mask_gts, corner_number, points_assigner_cfg):
    assert len(point_proposals) == len(mask_gts) == len(point_proposals_counter)
    if points_assigner_cfg.type == "CornerPointWithLine":
        return corner_anchor_points_assign_corner_with_line(point_proposals,
                                                              point_proposals_counter,
                                                              mask_gts,
                                                              corner_number,
                                                              points_assigner_cfg)
    else:
        raise NotImplementedError("Wrong points assigner")


def anchor_inside_flags(flat_anchors, valid_flags, img_shape,
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


def unmap_by_list(data, count, inds, fill=0):
    ret = np.full(count, fill, dtype=object)
    inds_np = (inds == 1).cpu().numpy()
    ret[inds_np] = data
    ret = ret.tolist()
    return ret
