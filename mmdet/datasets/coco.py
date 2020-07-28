import numpy as np
from pycocotools.coco import COCO
from functools import reduce
from typing import List

from .custom import CustomDataset
from .registry import DATASETS


@DATASETS.register_module
class CocoDataset(CustomDataset):

    CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic_light', 'fire_hydrant',
               'stop_sign', 'parking_meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat',
               'baseball_glove', 'skateboard', 'surfboard', 'tennis_racket',
               'bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot_dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted_plant', 'bed', 'dining_table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush')

    def load_annotations(self, ann_file):
        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.getCatIds()
        self.cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.cat_ids)
        }
        self.img_ids = self.coco.getImgIds()
        img_infos = []
        for i in self.img_ids:
            info = self.coco.loadImgs([i])[0]
            info['filename'] = info['file_name']
            img_infos.append(info)
        return img_infos

    def get_ann_info(self, idx):
        img_id = self.img_infos[idx]['id']
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.coco.loadAnns(ann_ids)
        return self._parse_ann_info(self.img_infos[idx], ann_info)

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        for i, img_info in enumerate(self.img_infos):
            if self.img_ids[i] not in ids_with_ann:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, seg_map. "masks" are raw annotations and not
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []

        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_polys = ann['segmentation']
                if self.clockwise_merge:
                    gt_polys = connect_polygons(ann['segmentation'])
                gt_masks_ann.append(gt_polys)

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann


def polygon_area(polygon, image_coordinate=True):
    assert len(polygon) >= 6
    x = polygon[0::2]
    y = polygon[1::2]
    area = 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    # In image, the original point is located at the top-left corner. So the area
    # should be multiplied by -1
    if image_coordinate:
        area *= -1.
    return area

def is_clockwise(polygon):
    area = polygon_area(polygon)
    return area < 0.

def connect_polygons(polygons: List[np.ndarray], mode="nearest"):

    clockwise_polygons = []
    for poly in polygons:
        poly = np.array(poly)
        if is_clockwise(poly):
            clockwise_polygons.append(poly)
        else:
            poly = poly.reshape(-1, 2)[::-1, :].reshape(-1)
            clockwise_polygons.append(poly)

    def connect_two_polygons(poly1, poly2):
        """
        All polygons are in clockwise.
        """
        poly1 = poly1.reshape(-1, 2)
        poly2 = poly2.reshape(-1, 2)
        match_dist_matrix = np.linalg.norm(poly1[:, None, :] - poly2, axis=2)
        min_dist = match_dist_matrix.min(axis=1)
        id1 = min_dist.argmin()
        id2 = match_dist_matrix[id1].argmin()

        poly1_1 = poly1[:id1+1]
        poly1_2 = poly1[id1:]
        poly2_1 = poly2[:id2+1]
        poly2_2 = poly2[id2:]

        poly_1 = np.concatenate([poly1_1, poly2_2], axis=0)
        poly_2 = np.concatenate([poly2_1, poly1_2], axis=0)
        new_poly = np.concatenate([poly_1, poly_2], axis=0)

        return new_poly
    
    connected_poly = reduce(connect_two_polygons, clockwise_polygons)
    
    return [connected_poly.reshape(-1).tolist()]
