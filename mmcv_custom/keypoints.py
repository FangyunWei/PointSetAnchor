import numpy as np
import torch

kpts_len = 17
def compute_oks(src_keypoints, src_roi, dst_keypoints, dst_roi):
    """Compute OKS for predicted keypoints wrt gt_keypoints.
    src_keypoints: 4xK
    src_roi: 4x1
    dst_keypoints: Nx4xK
    dst_roi: Nx4
    """

    sigmas = np.array([
        .26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87,
        .87, .89, .89]) / 10.0
    vars = (sigmas * 2)**2

    # area
    src_area = (src_roi[2] - src_roi[0] + 1) * (src_roi[3] - src_roi[1] + 1)

    # measure the per-keypoint distance if keypoints visible
    dx = dst_keypoints[:, 0, :] - src_keypoints[0, :]
    dy = dst_keypoints[:, 1, :] - src_keypoints[1, :]

    e = (dx**2 + dy**2) / vars / (src_area + np.spacing(1)) / 2
    e = np.sum(np.exp(-e), axis=1) / e.shape[1]

    return e

def center_oks_iou(g, d, a_g, a_d, vg, sigmas=None, in_vis_thre=None, type='oksiou'):
    keypoints, _ = get_keypoints()
    center = [keypoints.index('right_shoulder'), keypoints.index('left_shoulder')]
    if torch.mean(vg[center]) < 1:
        vg[center] = 0
    num_vis_point = 1
    if not isinstance(sigmas, np.ndarray):
        sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]) / 10.0
    vars = ((sigmas * 2) ** 2)[center[0]]
    xg = g[0::2][center]
    yg = g[1::2][center]
    #vg = g[2::3]
    xd = d[:, 0::2][:, center]
    yd = d[:, 1::2][:, center]
    #vd = d[n_d, 2::3]

    dx = torch.mean(xd, dim=-1).view(-1, 1) - torch.mean(xg)
    dy = torch.mean(yd, dim=-1).view(-1, 1) - torch.mean(yg)
    if type == 'oksiou':
        e = (dx ** 2 + dy ** 2) / d.new_tensor(vars) / ((a_g + a_d).view(-1, 1) / 2 + d.new_tensor(np.spacing(1))) / 2
    elif type == 'oksiof':
        e = (dx ** 2 + dy ** 2) / d.new_tensor(vars) / (a_g + np.spacing(1)) / 2
    else:
        assert 0, 'not impl'

    #if in_vis_thre is not None:
    #    ind = list(vg >= in_vis_thre) and list(vd >= in_vis_thre)
    #    e = e[ind]
    ious = torch.sum(torch.exp(-e) * vg[center][0], dim=-1) / num_vis_point
    return ious
def oks_iou(g, d, a_g, a_d, vg, sigmas=None, in_vis_thre=None, type='oksiou'):

    num_vis_point = torch.sum(vg)
    if num_vis_point == 0:
        return d.new_zeros((d.size()[0], ))
    if not isinstance(sigmas, np.ndarray):
        sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]) / 10.0
    vars = (sigmas * 2) ** 2
    xg = g[0::2]
    yg = g[1::2]
    #vg = g[2::3]
    xd = d[:, 0::2]
    yd = d[:, 1::2]
    #vd = d[n_d, 2::3]
    dx = xd - xg
    dy = yd - yg
    if type == 'oksiou':
        e = (dx ** 2 + dy ** 2) / d.new_tensor(vars) / ((a_g + a_d).view(-1, 1) / 2 + d.new_tensor(np.spacing(1))) / 2
    elif type == 'oksiof':
        e = (dx ** 2 + dy ** 2) / d.new_tensor(vars) / (a_g + np.spacing(1)) / 2
    else:
        assert 0, 'not impl'

    #if in_vis_thre is not None:
    #    ind = list(vg >= in_vis_thre) and list(vd >= in_vis_thre)
    #    e = e[ind]
    ious = torch.sum(torch.exp(-e) * vg, dim=-1) / num_vis_point
    #if torch.sum(g) == 0:
    #    iou_t = torch.sum(ious)
    #    pass
        #assert torch.sum(ious) == 0, 'gt with no visiable kpts should have 0 ious'
    # g_t, d_t, a_g_t, a_d_t = g.detach().cpu().numpy(), d.detach().cpu().numpy(), a_g.detach().cpu().numpy(), a_d.detach().cpu().numpy()
    # if not isinstance(sigmas, np.ndarray):
    #     sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]) / 10.0
    # vars_t = (sigmas * 2) ** 2
    # xg_t = g_t[0::2]
    # yg_t = g_t[1::2]
    # vg = g[2::3]
    # ious_test = np.zeros((d.shape[0]))
    # for n_d in range(0, d.shape[0]):
    #     xd_t = d_t[n_d, 0::2]
    #     yd_t = d_t[n_d, 1::2]
    #     #vd = d[n_d, 2::3]
    #     dx_t = xd_t - xg_t
    #     dy_t = yd_t - yg_t
    #     e_t = (dx_t ** 2 + dy_t ** 2) / vars_t / ((a_g_t + a_d_t[n_d]) / 2 + np.spacing(1)) / 2
    #     ious_test[n_d] = np.sum(np.exp(-e_t)) / e_t.shape[0] if e_t.shape[0] != 0 else 0.0
    #     ious_tensor = ious[n_d]
    return ious

def oks_iou_for(g, d, a_g, a_d, sigmas=None, in_vis_thre=None):
    if not isinstance(sigmas, np.ndarray):
        sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]) / 10.0
    vars = (sigmas * 2) ** 2
    xg = g[0::2]
    yg = g[1::2]
    #vg = g[2::3]
    ious = np.zeros((d.shape[0]))
    for n_d in range(0, d.shape[0]):
        xd = d[n_d, 0::2]
        yd = d[n_d, 1::2]
        #vd = d[n_d, 2::3]
        dx = xd - xg
        dy = yd - yg
        e = (dx ** 2 + dy ** 2) / vars / ((a_g + a_d[n_d]) / 2 + np.spacing(1)) / 2
        if in_vis_thre is not None:
            ind = list(vg >= in_vis_thre) and list(vd >= in_vis_thre)
            e = e[ind]
        ious[n_d] = np.sum(np.exp(-e)) / e.shape[0] if e.shape[0] != 0 else 0.0
    return ious

def oks_nms(kpts_db, thresh, sigmas=None, in_vis_thre=None):
    """
    greedily select boxes with high confidence and overlap with current maximum <= thresh
    rule out overlap >= thresh, overlap = oks
    :param kpts_db
    :param thresh: retain overlap < thresh
    :return: indexes to keep
    """
    if len(kpts_db) == 0:
        return []

    scores = np.array([kpts_db[i][-2] for i in range(len(kpts_db))])
    kpts = np.array([kpts_db[i][:-2].flatten() for i in range(len(kpts_db))])
    areas = np.array([kpts_db[i][-1] for i in range(len(kpts_db))])

    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        oks_ovr = oks_iou(kpts[i], kpts[order[1:]], areas[i], areas[order[1:]], sigmas, in_vis_thre)

        inds = np.where(oks_ovr <= thresh)[0]
        order = order[inds + 1]

    return keep
def get_keypoints():
    """Get the COCO keypoints and their left/right flip coorespondence map."""
    # Keypoints are not available in the COCO json for the test split, so we
    # provide them here.
    keypoints = [
        'nose',
        'left_eye',
        'right_eye',
        'left_ear',
        'right_ear',
        'left_shoulder',
        'right_shoulder',
        'left_elbow',
        'right_elbow',
        'left_wrist',
        'right_wrist',
        'left_hip',
        'right_hip',
        'left_knee',
        'right_knee',
        'left_ankle',
        'right_ankle'
    ]
    keypoint_flip_map = {
        'left_eye': 'right_eye',
        'left_ear': 'right_ear',
        'left_shoulder': 'right_shoulder',
        'left_elbow': 'right_elbow',
        'left_wrist': 'right_wrist',
        'left_hip': 'right_hip',
        'left_knee': 'right_knee',
        'left_ankle': 'right_ankle'
    }
    return keypoints, keypoint_flip_map

def flip_keypoints(keypoints, keypoint_flip_map, keypoint_coords, width):
    """Left/right flip keypoint_coords. keypoints and keypoint_flip_map are
    accessible from get_keypoints().
    """
    # flipped_kps = keypoint_coords.copy()
    flipped_kps = keypoint_coords.clone()
    for lkp, rkp in keypoint_flip_map.items():
        lid = keypoints.index(lkp)
        rid = keypoints.index(rkp)
        flipped_kps[:, lid, :] = keypoint_coords[:, rid, :]
        flipped_kps[:, rid, :] = keypoint_coords[:, lid, :]

    # Flip x coordinates
    flipped_kps[:, :, 0] = width - flipped_kps[:, :, 0] - 1
    # # Maintain COCO convention that if visibility == 0, then x, y = 0
    # inds = np.where(flipped_kps[:, 2, :] == 0)
    # flipped_kps[inds[0], 0, inds[1]] = 0
    return flipped_kps
