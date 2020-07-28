from mmdet.core import bbox_mapping_back
from ..registry import DETECTORS
from .single_stage import SingleStageDetector
import numpy as np
import cv2
import pycocotools.mask as mask_util
import torch

@DETECTORS.register_module
class PointSetAnchorDetector(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(PointSetAnchorDetector, self).__init__(backbone, neck, bbox_head, train_cfg,
                                   test_cfg, pretrained)

    # add property for --validate segm test
    @property
    def with_mask(self):
        return True

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_masks,
                      gt_bboxes_ignore=None):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        loss_inputs = outs + (gt_bboxes, gt_labels, gt_masks, img_metas, self.train_cfg)
        losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def simple_test(self, img, img_meta, rescale=False):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
        inference_res_list = self.bbox_head.get_inference_res(*bbox_inputs)
        results = [
            self.bbox_mask2result(det_bboxes, det_masks, det_labels, self.bbox_head.num_classes, img_meta[0])
            for det_bboxes, det_masks, det_labels in inference_res_list]

        bbox_results = results[0][0]
        mask_results = results[0][1]

        return bbox_results, mask_results


    def aug_test(self, imgs, img_metas, rescale=False):
        feats = self.extract_feats(imgs)
        aug_bboxes = []
        aug_masks = []
        aug_scores = []
        for x, img_meta in zip(feats, img_metas):
            outs = self.bbox_head(x)
            bbox_inputs = outs + (img_meta, self.test_cfg, False, False)
            det_bboxes, det_masks, det_scores = self.bbox_head.get_inference_res(*bbox_inputs)[0]
            aug_bboxes.append(det_bboxes)
            aug_masks.append(det_masks)
            aug_scores.append(det_scores)

        merged_bboxes, merged_masks, merged_scores = self.merge_aug_results(
                aug_bboxes, aug_masks, aug_scores, img_metas)
        det_bboxes, det_masks, det_labels = self.bbox_head.multiclass_nms_bbx_mask(
                merged_bboxes, merged_masks, merged_scores,
                self.test_cfg.score_thr, self.test_cfg.nms, self.test_cfg.max_per_img)

        if rescale:
            _det_bboxes = det_bboxes
            _det_masks = det_masks
        else:
            _det_bboxes = det_bboxes.clone()
            _det_masks = det_masks.clone()
            _det_bboxes[:, :4] *= img_metas[0][0]["scale_factor"]
            _det_masks *= img_metas[0][0]["scale_factor"]

        bbox_results, mask_results = self.bbox_mask2result(_det_bboxes, _det_masks, det_labels,
                self.bbox_head.num_classes, img_meta[0])

        return bbox_results, mask_results


    def merge_aug_results(self, aug_bboxes, aug_masks, aug_scores, img_metas):
        recovered_bboxes = []
        recovered_masks = []
        for bboxes, masks, img_info in zip(aug_bboxes, aug_masks, img_metas):
            img_shape = img_info[0]['img_shape']
            scale_factor = img_info[0]['scale_factor']
            flip = img_info[0]['flip']
            bboxes = bbox_mapping_back(bboxes, img_shape, scale_factor, flip)
            masks = bbox_mapping_back(masks, img_shape, scale_factor, flip)
            recovered_bboxes.append(bboxes)
            recovered_masks.append(masks)

        bboxes = torch.cat(recovered_bboxes, dim=0)
        masks = torch.cat(recovered_masks, dim=0)

        if aug_scores is None:
            return bboxes, masks
        else:
            scores = torch.cat(aug_scores, dim=0)
            return bboxes, masks, scores


    def bbox_mask2result(self, bboxes, masks, labels, num_classes, img_meta):
        """Convert detection results to a list of numpy arrays.

        Args:
            bboxes (Tensor): shape (n, 5)
            masks (Tensor): shape (n, template_point_number * 2)
            labels (Tensor): shape (n, )
            num_classes (int): class number, including background class

        Returns:
            list(ndarray): bbox results of each class
        """
        ori_shape = img_meta['ori_shape']
        img_h, img_w, _ = ori_shape
        mask_results = [[] for _ in range(num_classes - 1)]
        # convert to drawContours format
        masks_x = masks[:, 0::2]
        masks_y = masks[:, 1::2]
        masks = torch.stack((masks_x, masks_y), dim=2)

        for i in range(masks.shape[0]):
            im_mask = np.zeros((img_h, img_w), dtype=np.uint8)
            mask = [masks[i].unsqueeze(1).int().data.cpu().numpy()]
            im_mask = cv2.drawContours(im_mask, mask, -1,1,-1)
            rle = mask_util.encode(
                np.array(im_mask[:, :, np.newaxis], order='F'))[0]
            label = labels[i]
            mask_results[label].append(rle)

        if bboxes.shape[0] == 0:
            bbox_results = [
                np.zeros((0, 5), dtype=np.float32) for i in range(num_classes - 1)
            ]
            return bbox_results, mask_results
        else:
            bboxes = bboxes.cpu().numpy()
            labels = labels.cpu().numpy()
            bbox_results = [bboxes[labels == i, :] for i in range(num_classes - 1)]
            return bbox_results, mask_results