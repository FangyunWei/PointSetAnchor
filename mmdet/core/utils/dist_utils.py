from collections import OrderedDict

import torch.distributed as dist
from mmcv.runner import OptimizerHook
from torch._utils import (_flatten_dense_tensors, _take_tensors,
                          _unflatten_dense_tensors)


def _allreduce_coalesced(tensors, world_size, bucket_size_mb=-1):
    if bucket_size_mb > 0:
        bucket_size_bytes = bucket_size_mb * 1024 * 1024
        buckets = _take_tensors(tensors, bucket_size_bytes)
    else:
        buckets = OrderedDict()
        for tensor in tensors:
            tp = tensor.type()
            if tp not in buckets:
                buckets[tp] = []
            buckets[tp].append(tensor)
        buckets = buckets.values()

    for bucket in buckets:
        flat_tensors = _flatten_dense_tensors(bucket)
        dist.all_reduce(flat_tensors)
        flat_tensors.div_(world_size)
        for tensor, synced in zip(
                bucket, _unflatten_dense_tensors(flat_tensors, bucket)):
            tensor.copy_(synced)


def allreduce_grads(params, coalesce=True, bucket_size_mb=-1):
    grads = [
        param.grad.data for param in params
        if param.requires_grad and param.grad is not None
    ]
    world_size = dist.get_world_size()
    if coalesce:
        _allreduce_coalesced(grads, world_size, bucket_size_mb)
    else:
        for tensor in grads:
            dist.all_reduce(tensor.div_(world_size))


class DistOptimizerHook(OptimizerHook):

    def __init__(self, grad_clip=None, coalesce=True, bucket_size_mb=-1):
        self.grad_clip = grad_clip
        self.coalesce = coalesce
        self.bucket_size_mb = bucket_size_mb

    def after_train_iter(self, runner):
        runner.optimizer.zero_grad()
        runner.outputs['loss'].backward()
        allreduce_grads(runner.model.parameters(), self.coalesce,
                        self.bucket_size_mb)
        if self.grad_clip is not None:
            self.clip_grads(runner.model.parameters())
        runner.optimizer.step()


from mmcv.runner.hooks import TextLoggerHook
class TextLoggerAccHook(TextLoggerHook):
    def __init__(self, interval=10, ignore_last=True, reset_flag=False):
        super(TextLoggerAccHook, self).__init__(interval, ignore_last, reset_flag)

    def after_train_iter(self, runner):
        if self.every_n_inner_iters(runner, self.interval):
            # TODO(xiao): show avg of entire epoch
            runner.log_buffer.average()
        elif self.end_of_epoch(runner) and not self.ignore_last:
            # not precise but more stable
            # TODO(xiao): show avg of entire epoch
            runner.log_buffer.average()

        if runner.log_buffer.ready:
            self.log(runner)
            if self.reset_flag:
                runner.log_buffer.clear_output()

    def after_train_epoch(self, runner):
        runner.log_buffer.average()
        self.log(runner)
        runner.save_curve()
        if self.reset_flag:
            runner.log_buffer.clear_output()
