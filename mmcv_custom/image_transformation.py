import numpy as np
import math

PI = 3.14159265359


def identity_3x3mat():
    return np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)


def get_rotation_matrix_2d(r):
    # r in radian, counter-clockwise
    mat = np.array(
        [
            [math.cos(r),  math.sin(r), 0],
            [-math.sin(r), math.cos(r), 0],
            [0,            0,           1]
        ]
        , dtype=np.float64
    )
    return mat


def get_translation_matrix_2d(x, y):
    mat = np.array(
        [
            [1, 0, x],
            [0, 1, y],
            [0, 0, 1]
        ]
        , dtype=np.float64
    )
    return mat


def get_scale_matrix_2d(s):
    mat = np.array(
        [
            [s, 0, 0],
            [0, s, 0],
            [0, 0, 1]
        ]
        , dtype=np.float64
    )
    return mat


def get_shear_x_matrix_2d(k):
    # x_d = x_s + k * y_s
    # y_d = y_s
    mat = np.array(
        [
            [1, k, 0],
            [0, 1, 0],
            [0, 0, 1]
        ]
        , dtype=np.float64
    )
    return mat


def get_shear_y_matrix_2d(k):
    # x_d = x_s
    # y_d = y_s + k * x_s
    mat = np.array(
        [
            [1, 0, 0],
            [k, 1, 0],
            [0, 0, 1]
        ]
        , dtype=np.float64
    )
    return mat


def get_reflection_x_matrix_2d():
    mat = np.array(
        [
            [-1, 0, 0],
            [0,  1, 0],
            [0,  0, 1]
        ]
        , dtype=np.float64
    )
    return mat


def add_scale(mats, s):
    mats.append(get_scale_matrix_2d(s))


def add_reflect_x(mats, w):
    mats.append(get_reflection_x_matrix_2d())
    mats.append(get_translation_matrix_2d(w - 1, 0))


def __add_trans_and_centralize(mats, i_w, i_h, o_w, o_h, T, v):
    mats.append(get_translation_matrix_2d(-(i_w - 1) / 2.0, -(i_h - 1) / 2.0))
    mats.append(T(v))
    mats.append(get_translation_matrix_2d((o_w - 1) / 2.0, (o_h - 1) / 2.0))


def add_scale_and_centralize(mats, s, i_w, i_h, o_w, o_h):
    __add_trans_and_centralize(mats, i_w, i_h, o_w, o_h, get_scale_matrix_2d, s)


def add_rotate_and_centralize(mats, r, i_w, i_h, o_w, o_h):
    __add_trans_and_centralize(mats, i_w, i_h, o_w, o_h, get_rotation_matrix_2d, r)


def add_shear_x_and_centralize(mats, v, i_w, i_h, o_w, o_h):
    __add_trans_and_centralize(mats, i_w, i_h, o_w, o_h, get_shear_x_matrix_2d, v)


def add_shear_y_and_centralize(mats, v, i_w, i_h, o_w, o_h):
    __add_trans_and_centralize(mats, i_w, i_h, o_w, o_h, get_shear_y_matrix_2d, v)


def add_reflection_x_and_centralize(mats, v, i_w, i_h, o_w, o_h):
    assert i_w == o_w
    assert i_h == o_h
    if v > 0.5:
        mats.append(get_reflection_x_matrix_2d())
        mats.append(get_translation_matrix_2d(i_w - 1, 0))


def trans_point_2d(x, y, mat):
    p = np.array([[x], [y], [1]], dtype=np.float64)
    t = np.matmul(mat, p)
    assert t[2] == 1
    return t[0], t[1]


def trans_points_2d(x, y, mat):
    the_ones = np.ones(x.shape)
    p = np.stack([x, y, the_ones])
    t = np.matmul(mat, p)
    # assert t[2] == 1
    return t[0, :], t[1, :]


def get_composed_trans_matrix_2d(mats):
    # left multiplication
    mat = identity_3x3mat()
    for n in range(0, len(mats)):
        mat = np.matmul(mats[n], mat)
    return mat


import cv2


def transform_image(img, w, h, m):
    return cv2.warpAffine(img, m[0:2, :], (w, h))


def transform_image_sequentially(img, w, h, mats):
    m = get_composed_trans_matrix_2d(mats)
    return cv2.warpAffine(img, m[0:2, :], (w, h))


def generate_random_trans_images(imgs, org_width, org_height, width, height, scope, trans_type):
    if trans_type == "scale":
        T = add_scale_and_centralize
    elif trans_type == "rotation":
        T = add_rotate_and_centralize
    elif trans_type == "shear_x":
        T = add_shear_x_and_centralize
    elif trans_type == "shear_y":
        T = add_shear_y_and_centralize
    elif trans_type == "reflection_x":
        T = add_reflection_x_and_centralize
    else:
        assert 0, "error! unknown transformation type!"

    db_img = []
    db_tm = []
    for i in range(0, len(imgs)):
        if i % 100 == 0:
            print("\rgenerate random trans images: {} in {}. ".format(i, len(imgs)), end='')
        t = np.random.uniform(scope[0], scope[1])
        mats = [identity_3x3mat()]
        T(mats, t, org_width, org_height, width, height)
        m = get_composed_trans_matrix_2d(mats)
        db_img.append(transform_image(imgs[i], width, height, m))
        db_tm.append(m[0:2, :])
    print()
    db_img = np.asarray(db_img)
    db_tm = np.asarray(db_tm)
    return db_img, db_tm

# test_img = np.array(
#     [
#         [0, 1, 0, 1, 0],
#         [8, 3, 8, 6, 5],
#         [9, 1, 0, 2, 4],
#         [0, 1, 0, 1, 0],
#         [0, 1, 0, 1, 0],
#         [0, 1, 0, 1, 0],
#     ]
#     , dtype=np.uint8
# )
