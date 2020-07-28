import torch

from mmdet.ops.nms import nms_wrapper


def kpts_nms(multi_kpts,
             multi_scores,
             multi_areas,
             multi_vises,
             score_thr,
             nms_cfg,
             max_num=-1,
             multi_kpts_init=None,
             score_factors=None):
    """NMS for multi-class kpts.

    Args:
        kpts (Tensor): shape (n, #class*4) or (n, 4)
        scores (Tensor): shape (n, #class)
        score_thr (float): bbox threshold, kpts with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int): if there are more than max_num kpts after NMS,
            only top max_num will be kept.
        score_factors (Tensor): The factors multiplied to scores before
            applying NMS

    Returns:
        tuple: (kpts, labels), tensors of shape (k, 2 * 2 + 17 * 2 + 1 = 39) and (k, 1). Labels
            are 0-based.
    """
    boxpt_num = 2
    kpt_num = 17
    num_classes = multi_scores.shape[1]
    kpts, labels, vises = [], [], []
    if multi_kpts_init is not None:
        init_kpts = []

    nms_cfg_ = nms_cfg.copy()
    nms_type = nms_cfg_.pop('type', 'nms')
    nms_op = getattr(nms_wrapper, nms_type)
    for i in range(1, num_classes):
        cls_inds = multi_scores[:, i] > score_thr
        if not cls_inds.any():
            continue
        # get kpts and scores of this class, kpts is [4:34], without 2 corner points
        _kpts = multi_kpts[cls_inds, :]
        if multi_kpts_init is not None:
            _kpts_init = multi_kpts_init[cls_inds, :]
        _scores = multi_scores[cls_inds, i]
        _areas = multi_areas[cls_inds]
        _vises = multi_vises[cls_inds, :]
        if score_factors is not None:
            _scores *= score_factors[cls_inds]
        dets = torch.cat([_kpts, _scores[:, None], _areas.view(-1, 1)], dim=1)
        _, inds = nms_op(dets[..., boxpt_num * 2:], **nms_cfg_)
        cls_labels = multi_kpts.new_full(
            (inds.shape[0], ), i - 1, dtype=torch.long)
        # remove area
        kpts.append(dets[inds, :-1])
        vises.append(_vises[inds, :])
        labels.append(cls_labels)
        if multi_kpts_init is not None:
            init_kpts.append(_kpts_init[inds, :])
    if kpts:
        kpts = torch.cat(kpts)
        vises = torch.cat(vises)
        labels = torch.cat(labels)
        if multi_kpts_init is not None:
            init_kpts = torch.cat(init_kpts)
        if kpts.shape[0] > max_num:
            _, inds = kpts[:, -1].sort(descending=True)
            inds = inds[:max_num]
            kpts = kpts[inds]
            if multi_kpts_init is not None:
                init_kpts = init_kpts[inds]
            vises = vises[inds]
            labels = labels[inds]
    else:
        kpts = multi_kpts.new_zeros((0, (boxpt_num + kpt_num) * 2 + 1))
        if multi_kpts_init is not None:
            init_kpts = multi_kpts_init.new_zeros((0, (boxpt_num + kpt_num) * 2 + 1))
        vises = multi_kpts.new_zeros((0, kpt_num))
        labels = multi_kpts.new_zeros((0, ), dtype=torch.long)

    if multi_kpts_init is not None:
        return kpts, labels, vises, init_kpts
    return kpts, labels, vises


def kpts_nms_vis(multi_kpts,
                 multi_scores,
                 multi_areas,
                 multi_vises,
                 score_thr,
                 nms_cfg,
                 max_num=-1,
                 score_factors=None):
    """NMS for multi-class kpts.

    Args:
        kpts (Tensor): shape (n, #class*4) or (n, 4)
        scores (Tensor): shape (n, #class)
        score_thr (float): bbox threshold, kpts with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int): if there are more than max_num kpts after NMS,
            only top max_num will be kept.
        score_factors (Tensor): The factors multiplied to scores before
            applying NMS

    Returns:
        tuple: (kpts, labels), tensors of shape (k, 2 * 2 + 17 * 2 + 1 = 39) and (k, 1). Labels
            are 0-based.
    """
    boxpt_num = 2
    kpt_num = 17
    num_classes = multi_scores.shape[1]
    kpts, labels, vises = [], [], []
    nms_cfg_ = nms_cfg.copy()
    nms_type = nms_cfg_.pop('type', 'nms')
    nms_op = getattr(nms_wrapper, nms_type)
    for i in range(1, num_classes):
        cls_inds = multi_scores[:, i] > score_thr
        if not cls_inds.any():
            continue
        # get kpts and scores of this class, kpts is [4:34], without 2 corner points
        _kpts = multi_kpts[cls_inds, :]
        _scores = multi_scores[cls_inds, i]
        _areas = multi_areas[cls_inds]
        _vises = multi_vises[cls_inds, :]
        if score_factors is not None:
            _scores *= score_factors[cls_inds]
        dets = torch.cat([_kpts, _scores[:, None], _areas.view(-1, 1), _vises], dim=1)
        _, inds = nms_op(dets[..., boxpt_num * 2:], **nms_cfg_)
        cls_labels = multi_kpts.new_full(
            (inds.shape[0], ), i - 1, dtype=torch.long)
        # remove area
        kpts.append(dets[inds, :39])
        vises.append(dets[inds, 40:])
        labels.append(cls_labels)
    if kpts:
        kpts = torch.cat(kpts)
        vises = torch.cat(vises)
        labels = torch.cat(labels)
        if kpts.shape[0] > max_num:
            _, inds = kpts[:, -1].sort(descending=True)
            inds = inds[:max_num]
            kpts = kpts[inds]
            vises = vises[inds]
            labels = labels[inds]
    else:
        kpts = multi_kpts.new_zeros((0, (boxpt_num + kpt_num) * 2 + 1))
        vises = multi_kpts.new_zeros((0, kpt_num))
        labels = multi_kpts.new_zeros((0, ), dtype=torch.long)

    return kpts, labels, vises
