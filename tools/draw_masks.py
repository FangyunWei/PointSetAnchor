from mmdet.apis import init_detector, inference_detector, show_result
import mmcv
import os
config_file = './configs/templatepoints/templatepoints_baseline_r50_fpn_1x_4GPUs.py'
checkpoint_file = './experiments/templatepoints_baseline_r50_fpn_1x_4GPUs/epoch_12.pth'
test_num = 50
images_save_path = r'./experiments/VisualResults/templatepoints_baseline_r50_fpn_1x_4GPUs'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

images_root_path = './data/coco/val2017'

if not os.path.exists(images_save_path):
	os.makedirs(images_save_path)

for i, img in enumerate(os.listdir(images_root_path)):
	img_path = '{}/{}'.format(images_root_path,img)
	print('[{}/{}]'.format(i, test_num))
	out_file_path = '{}/{}'.format(images_save_path,img)
	result = inference_detector(model, img_path)
	show_result(img_path, result, model.CLASSES, score_thr=0.3, wait_time=0, show=False, out_file=out_file_path)
	if i == test_num:
		print('Done')
		break