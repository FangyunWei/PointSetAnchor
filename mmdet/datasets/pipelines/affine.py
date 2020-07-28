import numpy as np
import cv2


def norm_rot_angle(rot):
    norm_rot = rot
    while norm_rot > 180:
        norm_rot = norm_rot - 360
    while norm_rot <= -180:
        norm_rot = norm_rot + 360
    return norm_rot


def rotate_2d(pt_2d, rot_rad):
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)


def trans_point2d(pt_2d, trans):
    src_pt = np.array([pt_2d[0], pt_2d[1], 1.]).T
    dst_pt = np.dot(trans, src_pt)
    return dst_pt[0:2]


def trans_points_3d(_joints, trans, depth_scale):
    joints = _joints.copy()
    for n_jt in range(len(joints)):
        joints[n_jt, 0:2] = trans_point2d(joints[n_jt, 0:2], trans)
        joints[n_jt, 2] = joints[n_jt, 2] * depth_scale
    return joints


def fliplr_joints(_joints, _joints_vis, width, matched_parts):
    """
    flip coords
    :param _joints: numpy array, nJoints * dim, dim == 2 [x, y] or dim == 3  [x, y, z]
    :param _joints_vis: same as joints
    :param width: image width
    :param matched_parts: list of pairs
    :return:
    """
    joints = _joints.copy()
    joints_vis = _joints_vis.copy()
    joints[:, 0] = width - joints[:, 0] - 1
    # Change left-right parts
    for pair in matched_parts:
        joints[pair[0], :], joints[pair[1], :] = joints[pair[1], :], joints[pair[0], :].copy()
        joints_vis[pair[0], :], joints_vis[pair[1], :] = joints_vis[pair[1], :], joints_vis[pair[0], :].copy()
    return joints, joints_vis


def gen_affine_trans_from_box_cv(c_x, c_y, src_width, src_height, dst_width, dst_height, scale, rot, inv):
    """
    :param c_x, c_y, src_width, src_height: define a box
    :param dst_width, dst_height: target image size
    :param scale: augment image size, default 1.0
    :param rot: augment box rotation, default 0.0
    :param inv: False: image domain to patch domain. True: patch domain to image domain. Default False.
    :return:
    """
    # augment size with scale
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.array([c_x, c_y], dtype=np.float32)
    # augment rotation
    rot_rad = np.pi * rot / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    return trans


def gen_patch_image_from_box_cv(cvimg, c_x, c_y, bb_width, bb_height, patch_width, patch_height, do_flip, scale, rot):
    """
    :param cvimg: original image
    :param c_x, c_y, bb_width, bb_height: define a box
    :param patch_width, patch_height: target patch image size
    :param do_flip: flip augment
    :param scale: scale augment
    :param rot: rotation augment
    :return:
    """
    img = cvimg.copy()
    img_height, img_width, img_channels = img.shape
    if do_flip:
        img = img[:, ::-1, :]
        c_x = img_width - c_x - 1
    trans = gen_affine_trans_from_box_cv(c_x, c_y, bb_width, bb_height, patch_width, patch_height, scale, rot, False)
    img_patch = cv2.warpAffine(img, trans, (int(patch_width), int(patch_height)), flags=cv2.INTER_LINEAR)
    return img_patch, trans


def trans_coords_from_patch_to_org_2d(coords_in_patch, c_x, c_y, bb_width, bb_height, rot, patch_width, patch_height):
    coords_in_org = coords_in_patch.copy()
    trans = gen_affine_trans_from_box_cv(c_x, c_y, bb_width, bb_height, patch_width, patch_height, 1.0, rot, True)
    for p in range(coords_in_patch.shape[0]):
        coords_in_org[p, 0:2] = trans_point2d(coords_in_patch[p, 0:2], trans)
    return coords_in_org


def trans_coords_from_patch_to_org_3d(coords_in_patch, c_x, c_y, bb_width, bb_height, rot, patch_width, patch_height,
                                      depth_scale):
    coords_in_org = trans_coords_from_patch_to_org_2d(coords_in_patch, c_x, c_y, bb_width, bb_height, rot, patch_width,
                                                      patch_height)
    coords_in_org[:, 2] = coords_in_patch[:, 2] * depth_scale
    return coords_in_org

