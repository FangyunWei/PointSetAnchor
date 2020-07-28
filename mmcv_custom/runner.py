import os
import os.path as osp
import numpy as np
import logging
import pkgutil
from importlib import import_module
from collections import OrderedDict
import mmcv
from mmcv.runner.utils import obj_from_dict
from mmcv.runner.checkpoint import save_checkpoint
from mmcv.runner.checkpoint import load_url_dist, open_mmlab_model_urls, load_state_dict
import torch
from mmcv.runner import hooks
from .learning import plot_learning_curve


def load_checkpoint(model,
                    filename,
                    map_location=None,
                    strict=False,
                    logger=None):
    """Load checkpoint from a file or URI.

    Args:
        model (Module): Module to load checkpoint.
        filename (str): Either a filepath or URL or modelzoo://xxxxxxx.
        map_location (str): Same as :func:`torch.load`.
        strict (bool): Whether to allow different params for the model and
            checkpoint.
        logger (:mod:`logging.Logger` or None): The logger for error message.

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    # load checkpoint from modelzoo or file or url
    if filename.startswith('modelzoo://'):
        import torchvision
        model_urls = dict()
        for _, name, ispkg in pkgutil.walk_packages(
                torchvision.models.__path__):
            if not ispkg:
                _zoo = import_module('torchvision.models.{}'.format(name))
                if hasattr(_zoo, 'model_urls'):
                    _urls = getattr(_zoo, 'model_urls')
                    model_urls.update(_urls)
        model_name = filename[11:]
        checkpoint = load_url_dist(model_urls[model_name])
    elif filename.startswith('open-mmlab://'):
        model_name = filename[13:]
        checkpoint = load_url_dist(open_mmlab_model_urls[model_name])
    elif filename.startswith(('http://', 'https://')):
        checkpoint = load_url_dist(filename)
    else:
        if not osp.isfile(filename):
            raise IOError('{} is not a checkpoint file'.format(filename))
        checkpoint = torch.load(filename, map_location=map_location)
    # get state_dict from checkpoint
    if isinstance(checkpoint, OrderedDict):
        state_dict = checkpoint
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        raise RuntimeError(
            'No state_dict found in checkpoint file {}'.format(filename))
    # strip prefix of state_dict
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items()}
    # load state_dict

    if True:
        # TODO(xiao): init stage3 with stage2
        state_dict_ = state_dict.copy()
        for k, v in state_dict.items():
            if "extra_heads.0" in k:
                kk = k.replace("extra_heads.0", "extra_heads.1")
                state_dict_[kk] = v.clone()
        state_dict = state_dict_.copy()

    if hasattr(model, 'module'):
        load_state_dict(model.module, state_dict, strict, logger)
    else:
        load_state_dict(model, state_dict, strict, logger)
    return checkpoint


class Runner(mmcv.runner.Runner):
    """A training helper for PyTorch.

        Custom version of mmcv runner, overwrite init_optimizer method
    """

    def __init__(self,
                 model,
                 batch_processor,
                 optimizer=None,
                 work_dir=None,
                 log_level=logging.INFO,
                 logger=None):
        super(Runner, self).__init__(model, batch_processor, optimizer, work_dir, log_level, logger)
        self.loss_curve = OrderedDict()

    def save_checkpoint(self,
                        out_dir,
                        filename_tmpl='epoch_{}.pth',
                        save_optimizer=True,
                        meta=None):
        if meta is None:
            meta = dict(epoch=self.epoch + 1, iter=self.iter)
        else:
            meta.update(epoch=self.epoch + 1, iter=self.iter)

        filename = filename_tmpl.format(self.epoch + 1)
        filepath = osp.join(out_dir, filename)
        linkpath = osp.join(out_dir, 'latest.pth')
        optimizer = self.optimizer if save_optimizer else None
        save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
        save_checkpoint(self.model, linkpath, optimizer=optimizer, meta=meta)
        # use relative symlink
        # mmcv.symlink(filename, linkpath)

    def save_curve(self):
        # get info from current output
        for name, val in self.log_buffer.output.items():
            if name not in self.loss_curve:
                self.loss_curve[name] = []
            self.loss_curve[name].append(val)

        # save to json
        curvepath = osp.join(self.work_dir, 'latest_curve.json')
        # if os.path.exists(curvepath):
        #     os.remove(curvepath)
        mmcv.dump(self.loss_curve, curvepath)

        # plot curve
        for name, val in self.loss_curve.items():
            plot_learning_curve(val, val, self.work_dir, name)

    def auto_resume(self):
        linkname = osp.join(self.work_dir, 'latest.pth')
        if osp.exists(linkname):
            self.logger.info('latest checkpoint found')
            self.resume(linkname)

        curvepath = osp.join(self.work_dir, 'latest_curve.json')
        if osp.exists(curvepath):
            self.logger.info('latest_curve.json found')
            self.loss_curve = mmcv.load(curvepath)

    def gen_template(self, data_loader, **kwargs):
        res_root = "/data/coco/pose_sta/"
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(data_loader)
        for i, data_batch in enumerate(data_loader):
            self._inner_iter = i
            if i % (int(len(data_loader) / 100)) == 0:
                self.logger.info('Loading gt_keypoints ' + str(i) + ' in ' + str(len(data_loader)))
            d = data_batch['gt_keypoints'].data[0]
            d = torch.cat(d)
            d = d.detach().cpu().numpy()
            file_name = res_root + "{:0>8d}.npy".format(i)
            np.save(file_name, d)
            self._iter += 1
        self._epoch += 1

    def register_logger_hooks(self, log_config):
        log_interval = log_config['interval']
        for info in log_config['hooks']:
            if info.type == 'TextLoggerAccHook':
                logger_hook = TextLoggerAccHook(interval=log_interval)
            else:
                logger_hook = obj_from_dict(
                    info, hooks, default_args=dict(interval=log_interval))
            self.register_hook(logger_hook, priority='VERY_LOW')

    def resume(self, checkpoint, resume_optimizer=True,
               map_location='default'):
        if map_location == 'default':
            device_id = torch.cuda.current_device()
            checkpoint = self.load_checkpoint(
                checkpoint,
                map_location=lambda storage, loc: storage.cuda(device_id))
        else:
            checkpoint = self.load_checkpoint(
                checkpoint, map_location=map_location)

        if 'optimizer' in checkpoint and resume_optimizer:
            self._epoch = checkpoint['meta']['epoch']
            self._iter = checkpoint['meta']['iter']
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info('resumed epoch %d, iter %d', self.epoch, self.iter)

    def load_checkpoint(self, filename, map_location='cpu', strict=False):
        self.logger.info('load checkpoint from %s', filename)
        return load_checkpoint(self.model, filename, map_location, strict,
                               self.logger)
