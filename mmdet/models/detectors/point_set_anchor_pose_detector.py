from ..registry import DETECTORS
from .retinanet import RetinaNet
from mmdet.core import kpts2result, bbox_mapping_back, kpts_nms
from mmcv_custom.keypoints import get_keypoints, flip_keypoints
import copy
from .. import builder
import torch
import torch.nn as nn

from mmdet.models.anchor_heads.point_set_anchor_pose_head import TEMPLATE_POINTS_NUM, delta2template


def deep_copy_list_list_of_tensors(ttt):
    ddd = []
    for tt in ttt:
        dd = []
        for t in tt:
            dd.append(t.clone())
        ddd.append(dd)
    return ddd


@DETECTORS.register_module
class PointSetAnchorPoseDetector(RetinaNet):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 heat_head=None,
                 heat_branch_weight=1.0,
                 extra_stage_num=0,
                 stage2_oks_thre=0.85,
                 stage3_oks_thre=0.95,
                 use_predict_bbx=False,
                 heat_reg_group=False,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(PointSetAnchorPoseDetector, self).__init__(backbone, neck, bbox_head, train_cfg, test_cfg, pretrained)
        self.heat_head = builder.build_head(heat_head)
        self.heat_branch_weight = heat_branch_weight
        self.extra_stage_num = extra_stage_num
        self.use_out_scale = self.bbox_head.use_out_scale
        self.stage2_oks_thre = stage2_oks_thre
        self.stage3_oks_thre = stage3_oks_thre
        self.use_predict_bbx = use_predict_bbx
        self.heat_reg_group = heat_reg_group
        self.extra_heads = nn.ModuleList()

        for n_stage in range(self.extra_stage_num):
            self.extra_heads.append(builder.build_head(bbox_head))
            self.extra_heads[n_stage].init_weights()

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_keypoints=None
                      ):        
        x = self.extract_feat(img)
        all_anchor_list = []
        all_anchor_bbx_list = []
        anchor_list, anchor_bbx_list, valid_flag_list, anchor_zero_list, anchor_scale_list = \
            self.bbox_head.get_anchors([featmap.size()[-2:] for featmap in x], img_metas, device=x[0].device)
        all_anchor_list.append(anchor_list)
        all_anchor_bbx_list.append(anchor_bbx_list)
        outs = self.bbox_head(x, anchor_list, valid_flag_list, anchor_zero_list)
        loss_inputs = outs + (gt_labels, img_metas, self.train_cfg)
        losses = self.bbox_head.loss(
            *loss_inputs, gt_keypoints=gt_keypoints, gt_bboxes=gt_bboxes,
            out_anchor_list=deep_copy_list_list_of_tensors(anchor_list),
            out_anchor_bbx_list=deep_copy_list_list_of_tensors(anchor_bbx_list),
            out_valid_flag_list=deep_copy_list_list_of_tensors(valid_flag_list),
            out_anchor_scale_list=deep_copy_list_list_of_tensors(anchor_scale_list),
            gt_bboxes_ignore=gt_bboxes_ignore
        )

        if self.heat_head is not None:
            (_, heat_pred, offset) = self.heat_head(x)

            target = [self.heat_head.get_target(heat_pred_i, gt_keypoints, index) for (index, heat_pred_i) in
                      enumerate(heat_pred)]
            heat_loss = self.heat_head.loss([x[0] for x in target], [x[2] for x in target],
                                            [x[3] for x in target], heat_pred, offset)
            losses['heat_loss'] = [x * self.heat_branch_weight for x in heat_loss['heat_loss']]
            if 'offset_loss' in heat_loss.keys():
                losses['offset_loss'] = [x * self.heat_branch_weight for x in heat_loss['offset_loss']]

        for n_stage in range(self.extra_stage_num):
            bbox_inputs = outs + (heat_pred, offset, img_metas, self.test_cfg)
            anchor_list, anchor_bbx_list = self.bbox_head.get_bboxes(
                *bbox_inputs, rescale=False, do_nms=False, out_anchors=all_anchor_list[-1], out_bbx_anchors=all_anchor_bbx_list[-1], 
                out_anchors_scales=anchor_scale_list, use_heatmap=False, get_nextstage_anchor=True)
            all_anchor_list.append(anchor_list)
            all_anchor_bbx_list.append(anchor_bbx_list)
            outs = self.extra_heads[n_stage](x, anchor_list, valid_flag_list, anchor_zero_list)
            if n_stage == 0:
                train_cfg_copy = copy.deepcopy(self.train_cfg)
                train_cfg_copy['assigner']['pos_iou_thr'] = self.stage2_oks_thre
                train_cfg_copy['assigner']['neg_iou_thr'] = self.stage2_oks_thre - 0.1
            elif n_stage == 1:
                train_cfg_copy = copy.deepcopy(self.train_cfg)
                train_cfg_copy['assigner']['pos_iou_thr'] = self.stage3_oks_thre
                train_cfg_copy['assigner']['neg_iou_thr'] = self.stage3_oks_thre - 0.1
            else:
                assert 0
            loss_inputs = outs + (gt_labels, img_metas, train_cfg_copy)
            extra_losses = self.extra_heads[n_stage].loss(
                *loss_inputs, gt_keypoints=gt_keypoints, gt_bboxes=gt_bboxes,
                out_anchor_list=deep_copy_list_list_of_tensors(anchor_list),
                out_anchor_bbx_list=deep_copy_list_list_of_tensors(anchor_bbx_list),
                out_valid_flag_list=deep_copy_list_list_of_tensors(valid_flag_list),
                out_anchor_scale_list=deep_copy_list_list_of_tensors(anchor_scale_list),
                gt_bboxes_ignore=gt_bboxes_ignore
            )
            for k, v in extra_losses.items():
                losses['{}_stage{}'.format(k, n_stage)] = v

        return losses

    def simple_test(self, img, img_meta, rescale=False):
        x = self.extract_feat(img)

        all_anchor_list = []
        all_anchor_bbx_list = []
        anchor_list, anchor_bbx_list, valid_flag_list, anchor_zero_list, anchor_scale_list = \
            self.bbox_head.get_anchors([featmap.size()[-2:] for featmap in x], img_meta, device=x[0].device)
        all_anchor_list.append(anchor_list)
        all_anchor_bbx_list.append(anchor_bbx_list)

        outs = self.bbox_head(x, anchor_list, valid_flag_list, anchor_zero_list)

        if self.heat_head is not None:
            (_, heat_preds, offset) = self.heat_head(x)

        for n_stage in range(self.extra_stage_num):
            bbox_inputs = outs + (heat_preds, offset, img_meta, self.test_cfg)

            anchor_list, anchor_bbx_list = self.bbox_head.get_bboxes(
                *bbox_inputs, rescale=False, do_nms=False, out_anchors=all_anchor_list[-1], out_bbx_anchors=all_anchor_bbx_list[-1], 
                out_anchors_scales=anchor_scale_list, use_heatmap=False, get_nextstage_anchor=True)
            all_anchor_list.append(anchor_list)
            all_anchor_bbx_list.append(anchor_bbx_list)
            outs = self.extra_heads[n_stage](x, anchor_list, valid_flag_list, anchor_zero_list)

            
        bbox_inputs = outs + (heat_preds, offset, img_meta, self.test_cfg, False)
        
        bbox_list = self.bbox_head.get_bboxes(
            *bbox_inputs, do_nms=False, use_heatmap=self.heat_reg_group, out_anchors=all_anchor_list[-1], out_bbx_anchors=all_anchor_bbx_list[-1], out_anchors_scales=anchor_scale_list)
        bbox_results = [
            kpts2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results[0]

    def merge_aug_results(self, aug_bboxes, aug_poses, aug_scores, aug_areas, aug_vis, img_metas):
        recovered_bboxes = []
        recovered_poses = []
        recovered_areas = []
        for bboxes, poses, areas, img_info in zip(aug_bboxes, aug_poses, aug_areas, img_metas):
            img_shape = img_info[0]['img_shape']
            scale_factor = img_info[0]['scale_factor']
            flip = img_info[0]['flip']
            # assert not flip
            bboxes = bbox_mapping_back(bboxes, img_shape, scale_factor, flip)

            if flip:
                kpt_name, kpt_flip_map = get_keypoints()
                poses = poses.view(poses.shape[0], -1, 2)
                poses = flip_keypoints(kpt_name, kpt_flip_map, poses, img_shape[1])
                poses = poses.view(poses.shape[0], -1)
            poses = poses / scale_factor

            areas = areas / (scale_factor * scale_factor)
            recovered_bboxes.append(bboxes)
            recovered_poses.append(poses)
            recovered_areas.append(areas)

        bboxes = torch.cat(recovered_bboxes, dim=0)
        poses = torch.cat(recovered_poses, dim=0)
        areas = torch.cat(recovered_areas, dim=0)
        scores = torch.cat(aug_scores, dim=0)
        vis = torch.cat(aug_vis, dim=0)
        return bboxes, poses, areas, scores, vis

    def aug_test(self, imgs, img_metas, rescale=False):
        feats = self.extract_feats(imgs)
        aug_bbox_list, aug_pose_list, aug_score_list, aug_area_list, aug_vis_list = [],[],[],[],[]
        for i, (x, img_meta) in enumerate(zip(feats, img_metas)):
            
            all_anchor_list = []
            all_anchor_bbx_list = []
            anchor_list, anchor_bbx_list, valid_flag_list, anchor_zero_list, anchor_scale_list = \
                self.bbox_head.get_anchors([featmap.size()[-2:] for featmap in x], img_meta, device=x[0].device)
            all_anchor_list.append(anchor_list)
            all_anchor_bbx_list.append(anchor_bbx_list)
            outs = self.bbox_head(x, anchor_list, valid_flag_list, anchor_zero_list)

            if self.heat_head is not None:
                (_, heat_preds, offset) = self.heat_head(x)

            for n_stage in range(self.extra_stage_num):
                bbox_inputs = outs + (heat_preds, offset, img_meta, self.test_cfg)
                anchor_list, anchor_bbx_list = self.bbox_head.get_bboxes(
                    *bbox_inputs, rescale=False, do_nms=False, out_anchors=all_anchor_list[-1], out_bbx_anchors=all_anchor_bbx_list[-1], 
                    out_anchors_scales=anchor_scale_list, use_heatmap=False, get_nextstage_anchor=True)
                all_anchor_list.append(anchor_list)
                all_anchor_bbx_list.append(anchor_bbx_list)
                outs = self.extra_heads[n_stage](x, anchor_list, valid_flag_list, anchor_zero_list)
            
            bbox_inputs = outs + (heat_preds, offset, img_meta, self.test_cfg, False)
            bbox_list = self.bbox_head.get_bboxes(
                *bbox_inputs, do_nms=False, use_heatmap=self.heat_reg_group, out_anchors=all_anchor_list[-1], out_bbx_anchors=all_anchor_bbx_list[-1], out_anchors_scales=anchor_scale_list)
            assert len(bbox_list) == 1

            aug_bbox_list.append(bbox_list[0][0]) #3000,4
            aug_pose_list.append(bbox_list[0][1]) #3000,34
            aug_score_list.append(bbox_list[0][2]) #3000,2
            aug_area_list.append(bbox_list[0][3]) #3000
            aug_vis_list.append(bbox_list[0][4]) #3000,17

        merged_bboxes, merged_poses, merged_areas, merged_scores, merged_vis = \
            self.merge_aug_results(aug_bbox_list, aug_pose_list, aug_score_list, aug_area_list, aug_vis_list, img_metas)

        det_poses, det_labels, det_vises = kpts_nms(torch.cat([merged_bboxes, merged_poses], dim=-1), merged_scores,
                                                    merged_areas, merged_vis, self.test_cfg.score_thr,
                                                    self.test_cfg.nms, self.test_cfg.max_per_img)

        if rescale:
            _det_poses = det_poses
        else:
            _det_poses = det_poses.clone()
            _det_poses[:, :(TEMPLATE_POINTS_NUM + 2) * 2] *= img_metas[0][0]["scale_factor"]
        bbox_results = kpts2result(_det_poses, det_labels, self.bbox_head.num_classes)
        return bbox_results

    def show_result(self,
                    data,
                    result,
                    # img_norm_cfg,
                    dataset=None,
                    score_thr=0.3):
        assert 0
