import mmcv
import numpy as np
import torch


def bbox2delta(proposals, gt, means=[0, 0, 0, 0], stds=[1, 1, 1, 1]):
    assert proposals.size() == gt.size()

    proposals = proposals.float()
    gt = gt.float()
    px = (proposals[..., 0] + proposals[..., 2]) * 0.5
    py = (proposals[..., 1] + proposals[..., 3]) * 0.5
    pw = proposals[..., 2] - proposals[..., 0] + 1.0
    ph = proposals[..., 3] - proposals[..., 1] + 1.0

    gx = (gt[..., 0] + gt[..., 2]) * 0.5
    gy = (gt[..., 1] + gt[..., 3]) * 0.5
    gw = gt[..., 2] - gt[..., 0] + 1.0
    gh = gt[..., 3] - gt[..., 1] + 1.0

    dx = (gx - px) / pw
    dy = (gy - py) / ph
    dw = torch.log(gw / pw)
    dh = torch.log(gh / ph)
    deltas = torch.stack([dx, dy, dw, dh], dim=-1)

    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)
    deltas = deltas.sub_(means).div_(stds)

    return deltas


def delta2bbox(rois,
               deltas,
               means=[0, 0, 0, 0],
               stds=[1, 1, 1, 1],
               max_shape=None,
               wh_ratio_clip=16 / 1000):
    """
    Apply deltas to shift/scale base boxes.

    Typically the rois are anchor or proposed bounding boxes and the deltas are
    network outputs used to shift/scale those boxes.

    Args:
        rois (Tensor): boxes to be transformed. Has shape (N, 4)
        deltas (Tensor): encoded offsets with respect to each roi.
            Has shape (N, 4). Note N = num_anchors * W * H when rois is a grid
            of anchors. Offset encoding follows [1]_.
        means (list): denormalizing means for delta coordinates
        stds (list): denormalizing standard deviation for delta coordinates
        max_shape (tuple[int, int]): maximum bounds for boxes. specifies (H, W)
        wh_ratio_clip (float): maximum aspect ratio for boxes.

    Returns:
        Tensor: boxes with shape (N, 4), where columns represent
            tl_x, tl_y, br_x, br_y.

    References:
        .. [1] https://arxiv.org/abs/1311.2524

    Example:
        >>> rois = torch.Tensor([[ 0.,  0.,  1.,  1.],
        >>>                      [ 0.,  0.,  1.,  1.],
        >>>                      [ 0.,  0.,  1.,  1.],
        >>>                      [ 5.,  5.,  5.,  5.]])
        >>> deltas = torch.Tensor([[  0.,   0.,   0.,   0.],
        >>>                        [  1.,   1.,   1.,   1.],
        >>>                        [  0.,   0.,   2.,  -1.],
        >>>                        [ 0.7, -1.9, -0.5,  0.3]])
        >>> delta2bbox(rois, deltas, max_shape=(32, 32))
        tensor([[0.0000, 0.0000, 1.0000, 1.0000],
                [0.2817, 0.2817, 4.7183, 4.7183],
                [0.0000, 0.6321, 7.3891, 0.3679],
                [5.8967, 2.9251, 5.5033, 3.2749]])
    """
    means = deltas.new_tensor(means).repeat(1, deltas.size(1) // 4)
    stds = deltas.new_tensor(stds).repeat(1, deltas.size(1) // 4)
    denorm_deltas = deltas * stds + means
    dx = denorm_deltas[:, 0::4]
    dy = denorm_deltas[:, 1::4]
    dw = denorm_deltas[:, 2::4]
    dh = denorm_deltas[:, 3::4]
    max_ratio = np.abs(np.log(wh_ratio_clip))
    dw = dw.clamp(min=-max_ratio, max=max_ratio)
    dh = dh.clamp(min=-max_ratio, max=max_ratio)
    # Compute center of each roi
    px = ((rois[:, 0] + rois[:, 2]) * 0.5).unsqueeze(1).expand_as(dx)
    py = ((rois[:, 1] + rois[:, 3]) * 0.5).unsqueeze(1).expand_as(dy)
    # Compute width/height of each roi
    pw = (rois[:, 2] - rois[:, 0] + 1.0).unsqueeze(1).expand_as(dw)
    ph = (rois[:, 3] - rois[:, 1] + 1.0).unsqueeze(1).expand_as(dh)
    # Use exp(network energy) to enlarge/shrink each roi
    gw = pw * dw.exp()
    gh = ph * dh.exp()
    # Use network energy to shift the center of each roi
    gx = torch.addcmul(px, 1, pw, dx)  # gx = px + pw * dx
    gy = torch.addcmul(py, 1, ph, dy)  # gy = py + ph * dy
    # Convert center-xy/width/height to top-left, bottom-right
    x1 = gx - gw * 0.5 + 0.5
    y1 = gy - gh * 0.5 + 0.5
    x2 = gx + gw * 0.5 - 0.5
    y2 = gy + gh * 0.5 - 0.5
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1] - 1)
        y1 = y1.clamp(min=0, max=max_shape[0] - 1)
        x2 = x2.clamp(min=0, max=max_shape[1] - 1)
        y2 = y2.clamp(min=0, max=max_shape[0] - 1)
    bboxes = torch.stack([x1, y1, x2, y2], dim=-1).view_as(deltas)
    return bboxes


def bbox_flip(bboxes, img_shape):
    """Flip bboxes horizontally.

    Args:
        bboxes(Tensor or ndarray): Shape (..., 4*k)
        img_shape(tuple): Image shape.

    Returns:
        Same type as `bboxes`: Flipped bboxes.
    """
    if isinstance(bboxes, torch.Tensor):
        assert bboxes.shape[-1] % 4 == 0
        flipped = bboxes.clone()
        flipped[:, 0::4] = img_shape[1] - bboxes[:, 2::4] - 1
        flipped[:, 2::4] = img_shape[1] - bboxes[:, 0::4] - 1
        return flipped
    elif isinstance(bboxes, np.ndarray):
        return mmcv.bbox_flip(bboxes, img_shape)


def bbox_mapping(bboxes, img_shape, scale_factor, flip):
    """Map bboxes from the original image scale to testing scale"""
    new_bboxes = bboxes * scale_factor
    if flip:
        new_bboxes = bbox_flip(new_bboxes, img_shape)
    return new_bboxes


def bbox_mapping_back(bboxes, img_shape, scale_factor, flip):
    """Map bboxes from testing scale to original image scale"""
    new_bboxes = bbox_flip(bboxes, img_shape) if flip else bboxes
    new_bboxes = new_bboxes / scale_factor
    return new_bboxes


def bbox2roi(bbox_list):
    """Convert a list of bboxes to roi format.

    Args:
        bbox_list (list[Tensor]): a list of bboxes corresponding to a batch
            of images.

    Returns:
        Tensor: shape (n, 5), [batch_ind, x1, y1, x2, y2]
    """
    rois_list = []
    for img_id, bboxes in enumerate(bbox_list):
        if bboxes.size(0) > 0:
            img_inds = bboxes.new_full((bboxes.size(0), 1), img_id)
            rois = torch.cat([img_inds, bboxes[:, :4]], dim=-1)
        else:
            rois = bboxes.new_zeros((0, 5))
        rois_list.append(rois)
    rois = torch.cat(rois_list, 0)
    return rois


def roi2bbox(rois):
    bbox_list = []
    img_ids = torch.unique(rois[:, 0].cpu(), sorted=True)
    for img_id in img_ids:
        inds = (rois[:, 0] == img_id.item())
        bbox = rois[inds, 1:]
        bbox_list.append(bbox)
    return bbox_list


def bbox2result(bboxes, labels, num_classes):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (Tensor): shape (n, 5)
        labels (Tensor): shape (n, )
        num_classes (int): class number, including background class

    Returns:
        list(ndarray): bbox results of each class
    """
    if bboxes.shape[0] == 0:
        return [
            np.zeros((0, 5), dtype=np.float32) for i in range(num_classes - 1)
        ]
    else:
        bboxes = bboxes.cpu().numpy()
        labels = labels.cpu().numpy()
        return [bboxes[labels == i, :] for i in range(num_classes - 1)]


def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1] - 1)
        y1 = y1.clamp(min=0, max=max_shape[0] - 1)
        x2 = x2.clamp(min=0, max=max_shape[1] - 1)
        y2 = y2.clamp(min=0, max=max_shape[0] - 1)
    return torch.stack([x1, y1, x2, y2], -1)


def points2delta(point_proposals, bbx_proposals, assigned_points,
                 mean=0, std=1):
    '''
    :param point_proposals: template points
    :param bbx_proposals: anchors
    :param assigned_points:
    :param mean:
    :param std:
    :return: encoding results
            delta = ((GT Points - TemplatePoints)/AnchorSide - mean)/std
    '''
    assert len(bbx_proposals) == len(point_proposals) == len(assigned_points)

    point_proposals = point_proposals.float()
    assigned_mask_gts = assigned_points.float()
    bbx_proposals = bbx_proposals.float()

    # encoding mask regression target
    bw = bbx_proposals[:, 2] - bbx_proposals[:, 0] + 1.0
    bh = bbx_proposals[:, 3] - bbx_proposals[:, 1] + 1.0

    px = point_proposals[:, 0::2]
    py = point_proposals[:, 1::2]

    tx = assigned_mask_gts[:, 0::2]
    ty = assigned_mask_gts[:, 1::2]

    dx = (tx - px) / bw.unsqueeze(dim=-1)
    dy = (ty - py) / bh.unsqueeze(dim=-1)

    assigned_mask_gts[:, 0::2] = dx
    assigned_mask_gts[:, 1::2] = dy

    means = assigned_mask_gts.new_full(assigned_mask_gts.size(), mean)
    stds = assigned_mask_gts.new_full(assigned_mask_gts.size(), std)

    assigned_mask_gts = assigned_mask_gts.sub_(means).div_(stds)

    return assigned_mask_gts


def delta2points(point_proposals, bbx_proposals, mask_pred,
                 mean=0, std=1, max_shape=None, wh_ratio_clip=None):
    '''
    decoding:
    GT Points = (delta * std + mean) * AnchorSide + TemplatePoints
    '''
    bw = bbx_proposals[:, 2] - bbx_proposals[:, 0] + 1.0
    bh = bbx_proposals[:, 3] - bbx_proposals[:, 1] + 1.0
    wh = torch.stack([bw, bh], dim=1)
    whs = wh.repeat(1, point_proposals.size()[-1]//2)
    means = point_proposals.new_full(point_proposals.size(), mean)
    stds = point_proposals.new_full(point_proposals.size(), std)
    decoding_res = (mask_pred * stds + means) * whs + point_proposals
    if max_shape is not None:
        decoding_res_x = decoding_res[:, 0::2].clamp(min=0, max=max_shape[1] - 1)
        decoding_res_y = decoding_res[:, 1::2].clamp(min=0, max=max_shape[0] - 1)
        decoding_res[:, 0::2] = decoding_res_x
        decoding_res[:, 1::2] = decoding_res_y
    return decoding_res


def pointdist2distdelta(point_dists, bbx_proposals, point_counter,
                        mean=0, std=1):
    '''
    :param point_dists: template points dist to GT mask
    :param bbx_proposals: anchors
    :param mean:
    :param std:
    :return: encoding results
            delta = (point_dists/AnchorSide - mean)/std
    '''
    assert len(point_dists) == len(bbx_proposals) == len(point_counter)
    if len(point_dists.shape) == 1:
        point_dists = point_dists.unsqueeze(-1)
    if len(bbx_proposals.shape) == 1:
        bbx_proposals = bbx_proposals.unsqueeze(-1)
    if len(point_counter.shape) == 1:
        point_counter = point_counter.unsqueeze(-1)
    point_num = point_dists.shape[1]
    point_dists = point_dists.float()
    bbx_proposals = bbx_proposals.float()
    encoding_res = torch.zeros_like(point_dists)
    bw = bbx_proposals[:, 2] - bbx_proposals[:, 0] + 1.0
    bh = bbx_proposals[:, 3] - bbx_proposals[:, 1] + 1.0
    point_counter_index = torch.zeros_like(point_counter)
    tmp = point_counter.new_zeros(len(point_counter))
    for i in range(4):
        tmp += point_counter[:, i]
        point_counter_index[:, i] = tmp
    for w in bw.unique():
        idx = (bw == w).nonzero().squeeze(-1)
        cur_point_dists = point_dists[idx, :]
        cur_bw = bw[idx][0]
        cur_bh = bh[idx][0]
        cur_point_counter_index = point_counter_index[idx, :][0]
        cur_encoding_res = torch.zeros_like(cur_point_dists)
        # normalize dist of top points
        cur_encoding_res[:, 0:cur_point_counter_index[0]] =\
            cur_point_dists[:, 0:cur_point_counter_index[0]]/cur_bh
        # normalize dist of right points
        cur_encoding_res[:, cur_point_counter_index[0]:cur_point_counter_index[1]] =\
            cur_point_dists[:, cur_point_counter_index[0]:cur_point_counter_index[1]]/cur_bw
        # normalize dist of bottom points
        cur_encoding_res[:, cur_point_counter_index[1]:cur_point_counter_index[2]] = \
            cur_point_dists[:, cur_point_counter_index[1]:cur_point_counter_index[2]]/cur_bh
        # normalize dist of left points
        cur_encoding_res[:, cur_point_counter_index[2]:cur_point_counter_index[3]] = \
            cur_point_dists[:, cur_point_counter_index[2]:cur_point_counter_index[3]]/cur_bw
        encoding_res[idx, :] = cur_encoding_res

    means = encoding_res.new_full(encoding_res.size(), mean)
    stds = encoding_res.new_full(encoding_res.size(), std)
    encoding_res = encoding_res.sub_(means).div_(stds)

    return encoding_res


def distdelta2points(pred_point_dists, bbx_proposals, points, point_counter,
                     mean=0, std=1):
    '''
    :param point_dists: template points dist to GT mask
    :param bbx_proposals: anchors
    :param mean:
    :param std:
    :return:decoding results
            decoding_res = (delta * std + mean) * AnchorSide + points
            encoding:
            delta = (point_dists/AnchorSide - mean)/std
    '''
    assert len(pred_point_dists) == len(bbx_proposals) == len(points) == len(point_counter)
    point_num = pred_point_dists.shape[1]
    bbx_proposals = bbx_proposals.float()
    decoding_res = torch.zeros_like(points)
    bw = bbx_proposals[:, 2] - bbx_proposals[:, 0] + 1.0
    bh = bbx_proposals[:, 3] - bbx_proposals[:, 1] + 1.0
    point_counter_index = torch.zeros_like(point_counter)
    tmp = point_counter.new_zeros(len(point_counter))
    for i in range(4):
        tmp += point_counter[:, i]
        point_counter_index[:, i] = tmp
    for w in bw.unique():
        idx = (bw == w).nonzero().squeeze(-1)
        cur_pred_point_dists = pred_point_dists[idx, :]
        cur_points = points[idx, :]
        cur_bw = bw[idx][0]
        cur_bh = bh[idx][0]
        cur_point_counter_index = point_counter_index[idx, :][0]
        cur_decoding_res = decoding_res.new_zeros((len(idx), decoding_res.shape[1]))
        # top decoding
        cur_top_dists =\
            (cur_pred_point_dists[:, 0:cur_point_counter_index[0]] * std + mean) * cur_bh
        # same x
        cur_decoding_res[:, 0:cur_point_counter_index[0]*2:2] = \
            cur_points[:, 0:cur_point_counter_index[0]*2:2]
        # decode y
        cur_decoding_res[:, 1:cur_point_counter_index[0]*2:2] = \
            cur_points[:, 1:cur_point_counter_index[0]*2:2] + cur_top_dists
        # right decoding
        cur_right_dists =\
            (cur_pred_point_dists[:, cur_point_counter_index[0]:cur_point_counter_index[1]] * std + mean) * cur_bw
        # same y
        cur_decoding_res[:, cur_point_counter_index[0]*2 + 1:cur_point_counter_index[1]*2:2] = \
            cur_points[:, cur_point_counter_index[0]*2 + 1:cur_point_counter_index[1]*2:2]
        # decode x
        cur_decoding_res[:, cur_point_counter_index[0]*2:cur_point_counter_index[1]*2:2] = \
            cur_points[:, cur_point_counter_index[0]*2:cur_point_counter_index[1]*2:2] + cur_right_dists
        # bottom decoding
        cur_bottom_dists = \
            (cur_pred_point_dists[:, cur_point_counter_index[1]:cur_point_counter_index[2]] * std + mean) * cur_bh
        # same x
        cur_decoding_res[:, cur_point_counter_index[1]*2:cur_point_counter_index[2]*2:2] = \
            cur_points[:, cur_point_counter_index[1]*2:cur_point_counter_index[2]*2:2]
        # decode y
        cur_decoding_res[:, cur_point_counter_index[1]*2 + 1:cur_point_counter_index[2]*2:2] = \
            cur_points[:, cur_point_counter_index[1]*2 + 1:cur_point_counter_index[2]*2:2] + cur_bottom_dists
        # left decoding
        cur_left_dists = \
            (cur_pred_point_dists[:, cur_point_counter_index[2]:cur_point_counter_index[3]] * std + mean) * cur_bw
        # same y
        cur_decoding_res[:, cur_point_counter_index[2]*2 + 1:cur_point_counter_index[3]*2:2] = \
            cur_points[:, cur_point_counter_index[2]*2 + 1:cur_point_counter_index[3]*2:2]
        # decode x
        cur_decoding_res[:, cur_point_counter_index[2]*2:cur_point_counter_index[3]*2:2] = \
            cur_points[:, cur_point_counter_index[2]*2:cur_point_counter_index[3]*2:2] + cur_left_dists

        decoding_res[idx, :] = cur_decoding_res

    return decoding_res


def kpts2result(kpts, labels, num_classes):
    """Convert detection results to a list of numpy arrays.

    Args:
        kpts (Tensor): shape (n, 4 + 34 + 1)
        labels (Tensor): shape (n, )
        num_classes (int): should be 2

    Returns:
        list(ndarray): bbox results of ketpoints
    """
    if kpts.shape[0] == 0:
        return [np.zeros((0, 5), dtype=np.float32) for i in range(num_classes - 1)], [
            np.zeros((0, 35), dtype=np.float32) for i in range(num_classes - 1)]
    else:
        bboxes = torch.cat((kpts[:, :4], kpts[:, (kpts.shape[1] - 1):kpts.shape[1]]), dim=1)
        bboxes = bboxes.cpu().numpy()
        kpts = kpts[:, 4:].cpu().numpy()
        labels = labels.cpu().numpy()
        bbox_result = [bboxes[labels == i, :] for i in range(num_classes - 1)]
        keypoints_result = [kpts[labels == i, :] for i in range(num_classes - 1)]
        return bbox_result, keypoints_result


def pose2bbox_minmax(pts):
    pts_x = pts[:, 0::2]
    pts_y = pts[:, 1::2]

    bbox_left = pts_x.min(dim=-1, keepdim=True)[0]
    bbox_right = pts_x.max(dim=-1, keepdim=True)[0]
    bbox_up = pts_y.min(dim=-1, keepdim=True)[0]
    bbox_bottom = pts_y.max(dim=-1, keepdim=True)[0]
    bbox = torch.cat([bbox_left, bbox_up, bbox_right, bbox_bottom],
                     dim=-1)
    return bbox