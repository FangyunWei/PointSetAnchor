
# Point-Set Anchors for Object Detection, Instance Segmentation and Pose Estimation

The code for implementing **[Point-Set Anchors](https://arxiv.org/abs/2007.02846)**. The code is based on **[MMDetection](https://github.com/open-mmlab/mmdetection)**.

## Highlight
- A new object representation named Point-Set Anchors, which can be seen as
a generalization and extension of classical box anchors. Point-set anchors can
further provide informative features and better task-specific initializations
for shape regression.
- A network based on point-set anchors called PointSetNet, which is a modification of RetinaNet that simply replaces the anchor boxes with the proposed point-set anchors and also attaches a parallel regression branch.
Variants of this network are applied to object detection, human pose estimation, and also instance segmentation, for which the problem of defining specific regression targets is addressed.


## Performance
Object detection and snstance segmentation results on the MS COCO test-dev:

| Backbone  |Segm AP |Segm AP50 |Segm AP75 |Det AP |Det AP50 |Det AP75 |
|:---------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| ResNeXt-101-DCN  | 36.0   |  61.5   |  36.6  |  46.2   |  67.0   |  50.5   | 

Pose estimation results on the MS COCO test-dev:

| Backbone  |Pose AP |Pose AP50 |Pose AP75 |Pose APM |Pose APL |
|:---------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| HRNet-W48  | 68.7   |  89.9   |  76.3 |  64.8   |  75.3   |


## Quick start
### Requirements
1. Ubuntu 18.04
2. Python 3.6+
3. Pytorch 1.3+
4. mmcv 0.2.13 (other versions may not work)

### Installation
1. Clone this repo.
2. Follow the standard mmdetection installation step in [INSTALL.md](./docs/INSTALL.md) and organize the data.

### Models
Models can be downloaded in [Google Drive](https://drive.google.com/drive/folders/1GY4jAzBg8mBT7lvHY9433ERWlG5qT3wq?usp=sharing) and [Baidu Yun](https://pan.baidu.com/s/1TZ1uQLjKpQY9J8tk2b9bXQ)[Code:utwv].

HRNet-W48 imagenet pretrained model can be downloaded in [Google Drive](https://drive.google.com/file/d/1xk3tevawZ-XOK0y5DJi3TUsleM6B6e6p/view?usp=sharing) and [Baidu Yun](https://pan.baidu.com/s/1dZ6mfv8rybKdIqbBrtQo6w) [Code:vwkl].


### Data preparation
**For COCO data**, please download from [COCO download](http://cocodataset.org/#download), 2017 Train/Val is needed for COCO detection/instance segmentation/keypoints training and validation. 
Download and extract them under {ROOT}/data, and make them look like this:

    ${ROOT}
    |-- data
    `-- |-- coco
        `-- |-- annotations
            |   |-- person_keypoints_train2017.json
            |   `-- person_keypoints_val2017.json
            |   `-- instances_train2017.json
            |   `-- instances_val2017.json
            |-- train2017
            |-- val2017

### Training
The default configs are based on 8 GPUs per node. Please change the learning rate correspondingly for a machine with different number of GPUs.

Pose estimation (HRNet-W32)
- ```bash ./tools/dist_train.sh configs/point_set_anchor_pose/8GPUs_points_set_anchor_pose_HRNetw32_115E.py 8 ${YOUR_WORK_DIR}```

Pose estimation (HRNet-W48)
- ```bash ./tools/dist_train.sh configs/point_set_anchor_pose/8GPUs_points_set_anchor_pose_HRNetw48_115E.py 8 ${YOUR_WORK_DIR}```

Object detection and instance segmentation (ResNet50-DCN)
- ```bash ./tools/dist_train.sh configs/point_set_anchor_segm_det/8GPUs_point_set_anchor_segm_det_R50DCN_2x.py 8 ${YOUR_WORK_DIR}```

Object detection and instance segmentation (ResNeXt101-DCN)
- ```bash ./tools/dist_train.sh configs/point_set_anchor_segm_det/8GPUs_point_set_anchor_segm_det_RNeXt101DCN_2x.py 8 ${YOUR_WORK_DIR}```

### Testing
The default testings are based on 8 GPUs per node. Please change the GPU number correspondingly.

Pose estimation (HRNet-W32)
- ```bash tools/dist_test.sh configs/point_set_anchor_pose/8GPUs_points_set_anchor_pose_HRNetw32_115E.py ${YOUR_CHECKPOINT_PATH.pth} 8 --out {YOUR_OUTPUT_PATH.pkl} --eval keypoints```

Pose estimation (HRNet-W48)
- ```bash tools/dist_test.sh configs/point_set_anchor_pose/8GPUs_points_set_anchor_pose_HRNetw48_115E.py ${YOUR_CHECKPOINT_PATH.pth} 8 --out {YOUR_OUTPUT_PATH.pkl} --eval keypoints```

Object detection and instance segmentation (ResNet50-DCN)
- ```bash tools/dist_test.sh configs/point_set_anchor_segm_det/8GPUs_point_set_anchor_segm_det_R50DCN_2x.py ${YOUR_CHECKPOINT_PATH.pth} 8 --out {YOUR_OUTPUT_PATH.pkl} --eval bbox segm```

Object detection and instance segmentation (ResNeXt101-DCN)
- ```bash tools/dist_test.sh configs/point_set_anchor_segm_det/8GPUs_point_set_anchor_segm_det_RNeXt101DCN_2x.py ${YOUR_CHECKPOINT_PATH.pth} 8 --out {YOUR_OUTPUT_PATH.pkl} --eval bbox segm```


## Citation

Please consider citing our paper in your publications if the project helps your research. BibTeX reference is as follows.

```
@article{wei2020point,
  title={Point-Set Anchors for Object Detection, Instance Segmentation and Pose Estimation},
  author={Wei, Fangyun and Sun, Xiao and Li, Hongyang and Wang, Jingdong and Lin, Stephen},
  journal={arXiv preprint arXiv:2007.02846},
  year={2020}
}
```
