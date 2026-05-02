/*
 * Copyright (c) 2024 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef FLASHINFER_ATTENTION_VARIANTS_CUH_
#define FLASHINFER_ATTENTION_VARIANTS_CUH_
#include <cuda_runtime.h>

#include <cstdint>
#include <type_traits>

#include "../math.cuh"
#include "../utils.cuh"
#include "variant_helper.cuh"

namespace flashinfer {

DEFINE_HAS_MEMBER(maybe_mask_indptr)
DEFINE_HAS_MEMBER(num_kv_heads)

template <bool use_custom_mask, bool use_sliding_window, bool use_logits_soft_cap, bool use_alibi,
          bool use_router = false>
struct DefaultAttention : AttentionVariantBase {
  static constexpr bool use_softmax = true;

  uint8_t* custom_mask_ptr;
  uint32_t qo_len, kv_len;
  uint32_t window_left;
  uint32_t router_sink_size;
  float sm_scale_log2;
  float soft_cap_pre_tanh_scale;
  uint8_t* maybe_router;
  uint32_t num_kv_heads;
  uint64_t router_stride_n;
  uint64_t router_stride_h;
  bool router_is_aha_gate;

  // Create closure
  template <typename Params>
  __device__ __host__ DefaultAttention(const Params& params, uint32_t batch_idx,
                                       uint8_t* smem_ptr) {
    qo_len = params.get_qo_len(batch_idx);
    kv_len = params.get_kv_len(batch_idx);
    if constexpr (use_logits_soft_cap) {
      soft_cap_pre_tanh_scale = params.sm_scale * math::ptx_rcp(params.logits_soft_cap);
      sm_scale_log2 = math::log2e * params.logits_soft_cap;
    } else {
      if constexpr (use_alibi) {
        sm_scale_log2 = math::log2e;
      } else {
        sm_scale_log2 = params.sm_scale * math::log2e;
      }
    }
    if constexpr (use_custom_mask) {
      if constexpr (has_maybe_mask_indptr_v<Params>) {
        custom_mask_ptr = params.maybe_custom_mask + params.maybe_mask_indptr[batch_idx];
      } else {
        custom_mask_ptr = params.maybe_custom_mask;
      }
    }
    window_left = (params.window_left >= 0) ? params.window_left : kv_len;
    router_sink_size = 0;
    if constexpr (use_router) {
      maybe_router = params.maybe_router;
      router_is_aha_gate = false;
      if constexpr (has_num_kv_heads_v<Params>) {
        num_kv_heads = params.num_kv_heads;
      } else {
        num_kv_heads = params.paged_kv.num_heads;
      }
      if constexpr (has_router_sink_size_v<Params>) {
        router_sink_size = params.router_sink_size;
      }
      if constexpr (has_router_is_aha_gate_v<Params>) {
        router_is_aha_gate = params.router_is_aha_gate;
      }
      router_stride_n = num_kv_heads;
      router_stride_h = 1;
      if constexpr (has_router_stride_n_v<Params>) {
        router_stride_n = static_cast<uint64_t>(params.router_stride_n);
      }
      if constexpr (has_router_stride_h_v<Params>) {
        router_stride_h = static_cast<uint64_t>(params.router_stride_h);
      }
    }
  }

  REGISTER_LOGITS_TRANSFORM(params, logits, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx, {
    if constexpr (use_alibi) {
      logits = logits * params.sm_scale +
               params.maybe_alibi_slopes[qo_head_idx] * float(int(kv_idx) - int(qo_idx));
    }
    if constexpr (use_logits_soft_cap) {
      logits = float(math::tanh(logits * soft_cap_pre_tanh_scale));
    }
    return logits;
  })

  REGISTER_LOGITS_MASK(params, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx, {
    bool mask = true;
    if constexpr (use_custom_mask) {
      if (qo_idx >= qo_len || kv_idx >= kv_len) {
        mask = false;
      } else {
        const uint64_t offset = static_cast<uint64_t>(qo_idx) * kv_len + kv_idx;
        mask &= ((custom_mask_ptr[offset / 8] >> (offset % 8)) & 1);
      }
    }
    if constexpr (use_sliding_window && !use_router) {
      mask &= (kv_idx + qo_len + window_left >= kv_len + qo_idx);
    }
    if constexpr (use_router) {
      bool route_value;
      if constexpr (has_q_indptr_v<Params>) {
        if (router_is_aha_gate) {
          route_value =
              maybe_router[static_cast<uint64_t>(params.q_indptr[batch_idx] + qo_idx) *
                               router_stride_n +
                           static_cast<uint64_t>(kv_head_idx) * router_stride_h];
        } else {
          route_value = maybe_router[static_cast<uint64_t>(batch_idx) * router_stride_n +
                                     static_cast<uint64_t>(kv_head_idx) * router_stride_h];
        }
      } else {
        route_value = maybe_router[static_cast<uint64_t>(batch_idx) * router_stride_n +
                                   static_cast<uint64_t>(kv_head_idx) * router_stride_h];
      }
      bool head_uses_local = router_is_aha_gate ? !route_value : route_value;
      if (head_uses_local) {
        bool in_sink = kv_idx < router_sink_size;
        bool in_recent = kv_idx + qo_len + window_left >= kv_len + qo_idx;
        mask &= (in_sink || in_recent);
      }
    }
    return mask;
  })
};

};  // namespace flashinfer

#endif  // FLASHINFER_ATTENTION_VARIANTS_CUH_
