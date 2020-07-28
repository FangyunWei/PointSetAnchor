// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include <torch/extension.h>
#include <math.h>
#include <iostream>
#include <fstream>

#define KPTS_LEN 17
#define EPS 2.2204e-16

template <typename scalar_t>
inline scalar_t oks_iou(const scalar_t* kpts, int64_t _i, int64_t _j){

    //using scalar_t = float;
    
    scalar_t e[KPTS_LEN] = {0.0};
    scalar_t e_sum = 0.0;
    scalar_t ious = 0.0;
    int64_t i_shift = _i * (KPTS_LEN * 2 + 2);
    int64_t j_shift = _j * (KPTS_LEN * 2 + 2);

    scalar_t sigma[KPTS_LEN] = {0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089};
    for(int i = 0;i < KPTS_LEN; ++i){
        auto vars = 4 * sigma[i] * sigma[i];
        auto x_kpts_g = kpts[i_shift + i * 2];
        auto y_kpts_g = kpts[i_shift + i * 2 + 1];

        auto x_kpts_d = kpts[j_shift + i * 2];
        auto y_kpts_d = kpts[j_shift + i * 2 + 1];
        auto dx = x_kpts_d - x_kpts_g;
        auto dy = y_kpts_d - y_kpts_g;
        e[i] = (dx * dx + dy * dy) / vars / ((kpts[i_shift + KPTS_LEN * 2 + 1] + kpts[j_shift + KPTS_LEN * 2 + 1]) / 2  + EPS) / 2; 
    }
    for(int i = 0;i < KPTS_LEN; ++i){
            e_sum += exp(-1 * e[i]);
    }
    ious = e_sum / KPTS_LEN;
    
    return ious;
}

template <typename scalar_t>
at::Tensor oks_nms_cpu_kernel(const at::Tensor& kpts, 
                              const float overlap_thresh, 
                              const at::Tensor& sigmas) {
  AT_ASSERTM(!kpts.type().is_cuda(), "kpts must be a CPU tensor");

  if (kpts.numel() == 0) {
    return at::empty({0}, kpts.options().dtype(at::kLong).device(at::kCPU));
  }
  //using scalar_t = float;
  //scalar_t* kpts_s = kpts.data<scalar_t>();

  auto scores = kpts.select(1, KPTS_LEN * 2).contiguous();
  auto order_t = std::get<1>(scores.sort(0, /* descending=*/true));

  auto nkpts = kpts.size(0);
  at::Tensor suppressed_t =
      at::zeros({nkpts}, kpts.options().dtype(at::kByte).device(at::kCPU));

  auto suppressed = suppressed_t.data<uint8_t>();
  auto order = order_t.data<int64_t>();
  scalar_t* kpts_s = kpts.data<scalar_t>();
  for (int64_t _i = 0; _i < nkpts; _i++) {
    auto i = order[_i];
    if (suppressed[i] == 1) continue;

    for (int64_t _j = _i + 1; _j < nkpts; _j++) {
      auto j = order[_j];
      if (suppressed[j] == 1) continue;
      float oks_iou_gd =  oks_iou<scalar_t>(kpts_s, i, j);
      if (oks_iou_gd >= overlap_thresh) suppressed[j] = 1;
    }
  }
  return at::nonzero(suppressed_t == 0).squeeze(1);
}

at::Tensor oks_nms(const at::Tensor& kpts, 
                              const float overlap_thresh, 
                              const at::Tensor& sigmas) {
  at::Tensor result;
  AT_DISPATCH_FLOATING_TYPES(kpts.type(), "oks_nms", [&] {
    result = oks_nms_cpu_kernel<scalar_t>(kpts, overlap_thresh, sigmas); 
  });
  return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("oks_nms", &oks_nms, "oks non-maximum suppression");
}
