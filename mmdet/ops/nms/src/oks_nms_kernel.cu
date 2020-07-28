// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THC.h>
#include <THC/THCDeviceUtils.cuh>

#include <vector>
#include <iostream>
#include <math.h>
//17 key points * 2  +  1 scores and 1 areas
#define KPTS_LEN 17
#define EPS 2.2204e-16
int const threadsPerBlock = sizeof(unsigned long long) * 8;

__device__ inline float devIoU(float const * const kpts_g, float const * const kpts_d, float const * const sigmas) {

    float e[KPTS_LEN] = {0.0};
    float e_sum = 0.0;
    float ious = 0.0;

    float sigma[KPTS_LEN] = {0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089};

    for(int i = 0;i < KPTS_LEN; ++i){
        float vars = 4 * sigma[i] * sigma[i];
        float x_kpts_g = kpts_g[i * 2];
        float y_kpts_g = kpts_g[i * 2 + 1];

        float x_kpts_d = kpts_d[i * 2];
        float y_kpts_d = kpts_d[i * 2 + 1];
        float dx = x_kpts_d - x_kpts_g;
        float dy = y_kpts_d - y_kpts_g;
        e[i] = (dx * dx + dy * dy) / vars / ((kpts_g[KPTS_LEN * 2 + 1] + kpts_d[KPTS_LEN * 2 + 1]) / 2  + EPS) / 2; 
    }
    for(int i = 0;i < KPTS_LEN; ++i){
            e_sum += exp(-1 * e[i]);
    }
    ious = e_sum / KPTS_LEN;
    
    return ious;
}
__global__ void oks_nms_kernel(const int n_kpts, const float overlap_thresh, 
                               const float *dev_kpts, const float *sigmas, unsigned long long *dev_mask) {
  const int row_start = blockIdx.y;
  const int col_start = blockIdx.x;

  // if (row_start > col_start) return;

  const int row_size =
        min(n_kpts - row_start * threadsPerBlock, threadsPerBlock);
  const int col_size =
        min(n_kpts - col_start * threadsPerBlock, threadsPerBlock);

  __shared__ float block_kpts[threadsPerBlock * (KPTS_LEN * 2 + 2)];
  if (threadIdx.x < col_size) {
      for(int i = 0; i < (KPTS_LEN * 2 + 2); ++i){
        block_kpts[threadIdx.x * (KPTS_LEN * 2 + 2) + i] =
            dev_kpts[(threadsPerBlock * col_start + threadIdx.x) * (KPTS_LEN * 2 + 2) + i];
      }
  }
  __syncthreads();

  if (threadIdx.x < row_size) {
    const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
    const float *cur_kpt = dev_kpts + cur_box_idx * (KPTS_LEN * 2 + 2);
    int i = 0;
    unsigned long long t = 0;
    int start = 0;
    if (row_start == col_start) {
      start = threadIdx.x + 1;
    }
    for (i = start; i < col_size; i++) {
      if (devIoU(cur_kpt, block_kpts + i * (KPTS_LEN * 2 + 2), sigmas) > overlap_thresh) {
        t |= 1ULL << i;
      }
    }
    const int col_blocks = THCCeilDiv(n_kpts, threadsPerBlock);
    dev_mask[cur_box_idx * col_blocks + col_start] = t;
  }
}

// kpts is a N x 19 tensor, 18th is scores and 19th is areas
at::Tensor oks_nms_cuda(const at::Tensor kpts, 
                              float overlap_thresh,
                         const at::Tensor sigmas) {
  using scalar_t = float;
  AT_ASSERTM(kpts.type().is_cuda(), "kpts must be a CUDA tensor");
  auto scores = kpts.select(1, KPTS_LEN * 2);
  auto order_t = std::get<1>(scores.sort(0, /* descending=*/true));

  auto kpts_sorted = kpts.index_select(0, order_t);

  int kpts_num = kpts.size(0);

  const int col_blocks = THCCeilDiv(kpts_num, threadsPerBlock);

  scalar_t* kpts_dev = kpts_sorted.data<scalar_t>();
  scalar_t* sigmas_dev = kpts_sorted.data<scalar_t>();

  THCState *state = at::globalContext().lazyInitCUDA(); // TODO replace with getTHCState

  unsigned long long* mask_dev = NULL;
  //THCudaCheck(THCudaMalloc(state, (void**) &mask_dev,
  //                      kpts_num * col_blocks * sizeof(unsigned long long)));

  mask_dev = (unsigned long long*) THCudaMalloc(state, kpts_num * col_blocks * sizeof(unsigned long long));

  dim3 blocks(THCCeilDiv(kpts_num, threadsPerBlock),
              THCCeilDiv(kpts_num, threadsPerBlock));
  dim3 threads(threadsPerBlock);
  oks_nms_kernel<<<blocks, threads>>>(kpts_num, 
                                  overlap_thresh,  
                                      kpts_dev, sigmas_dev,
                                  mask_dev);


  std::vector<unsigned long long> mask_host(kpts_num * col_blocks);

  THCudaCheck(cudaMemcpy(&mask_host[0],
                        mask_dev,
                        sizeof(unsigned long long) * kpts_num * col_blocks,
                        cudaMemcpyDeviceToHost));

  std::vector<unsigned long long> remv(col_blocks);
  memset(&remv[0], 0, sizeof(unsigned long long) * col_blocks);

  at::Tensor keep = at::empty({kpts_num}, kpts.options().dtype(at::kLong).device(at::kCPU));
  int64_t* keep_out = keep.data<int64_t>();

  int num_to_keep = 0;
  for (int i = 0; i < kpts_num; i++) {
    int nblock = i / threadsPerBlock;
    int inblock = i % threadsPerBlock;

    if (!(remv[nblock] & (1ULL << inblock))) {
      keep_out[num_to_keep++] = i;
      unsigned long long *p = &mask_host[0] + i * col_blocks;
      for (int j = nblock; j < col_blocks; j++) {
        remv[j] |= p[j];
      }
    }
  }

  THCudaFree(state, mask_dev);
  // TODO improve this part
  return std::get<0>(order_t.index({
                       keep.narrow(/*dim=*/0, /*start=*/0, /*length=*/num_to_keep).to(
                         order_t.device(), keep.scalar_type())
                     }).sort(0, false));
}
