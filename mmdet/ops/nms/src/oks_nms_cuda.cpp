// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include <torch/extension.h>

#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")

at::Tensor oks_nms_cuda(const at::Tensor kpts, float oks_nms_overlap_thresh, const at::Tensor sigmas);

at::Tensor oks_nms(const at::Tensor& kpts, const float threshold, const at::Tensor& sigmas) {
  CHECK_CUDA(kpts);
  if (kpts.numel() == 0)
    return at::empty({0}, kpts.options().dtype(at::kLong).device(at::kCPU));
  return oks_nms_cuda(kpts, threshold, sigmas);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("oks_nms", &oks_nms, "oks non-maximum suppression");
}
