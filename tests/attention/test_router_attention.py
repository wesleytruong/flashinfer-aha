"""
Copyright (c) 2024 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import pytest
import torch

import flashinfer


def _make_paged_kv(batch_size, kv_len, num_kv_heads, head_dim, page_size, device="cuda:0"):
    """Helper to create paged KV cache and metadata."""
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    k_data = torch.randn(
        total_num_pages, page_size, num_kv_heads, head_dim,
        dtype=torch.float16, device=device,
    )
    v_data = torch.randn(
        total_num_pages, page_size, num_kv_heads, head_dim,
        dtype=torch.float16, device=device,
    )
    kv_indptr = (
        torch.arange(0, batch_size + 1, device=device, dtype=torch.int32) * num_pages_per_seq
    )
    kv_indices = torch.arange(0, total_num_pages, device=device, dtype=torch.int32)
    kv_last_page_len = torch.full(
        (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32, device=device,
    )
    return k_data, v_data, kv_indptr, kv_indices, kv_last_page_len


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("kv_len", [199, 1999])
@pytest.mark.parametrize("window_left", [33, 533])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("num_qo_heads", [4])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("page_size", [16])
def test_batch_decode_router_all_zeros(
    batch_size, kv_len, window_left, num_kv_heads, num_qo_heads, head_dim, page_size,
):
    """All-zeros router (all full attention) should match standard full attention."""
    q = torch.randn(batch_size, num_qo_heads, head_dim, dtype=torch.float16, device="cuda:0")
    k_data, v_data, kv_indptr, kv_indices, kv_last_page_len = _make_paged_kv(
        batch_size, kv_len, num_kv_heads, head_dim, page_size,
    )
    router = torch.zeros(batch_size, num_kv_heads, dtype=torch.uint8, device="cuda:0")

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device="cuda:0")

    # Reference: standard full attention (no sliding window)
    wrapper_ref = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace_buffer, "NHD")
    wrapper_ref.plan(
        kv_indptr, kv_indices, kv_last_page_len,
        num_qo_heads, num_kv_heads, head_dim, page_size,
    )
    o_ref = wrapper_ref.run(q, (k_data, v_data))

    # Router: all zeros = all full attention, window_left is set but ignored for full-attn heads
    wrapper_router = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer, "NHD", use_router=True,
    )
    wrapper_router.plan(
        kv_indptr, kv_indices, kv_last_page_len,
        num_qo_heads, num_kv_heads, head_dim, page_size,
        window_left=window_left,
    )
    o_router = wrapper_router.run(q, (k_data, v_data), router=router)

    torch.testing.assert_close(o_router.cpu(), o_ref.cpu(), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("kv_len", [199, 1999])
@pytest.mark.parametrize("window_left", [33, 533])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("num_qo_heads", [4])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("page_size", [16])
def test_batch_decode_router_all_ones(
    batch_size, kv_len, window_left, num_kv_heads, num_qo_heads, head_dim, page_size,
):
    """All-ones router (all SWA) should match standard SWA with same window_left."""
    q = torch.randn(batch_size, num_qo_heads, head_dim, dtype=torch.float16, device="cuda:0")
    k_data, v_data, kv_indptr, kv_indices, kv_last_page_len = _make_paged_kv(
        batch_size, kv_len, num_kv_heads, head_dim, page_size,
    )
    router = torch.ones(batch_size, num_kv_heads, dtype=torch.uint8, device="cuda:0")

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device="cuda:0")

    # Reference: standard SWA
    wrapper_ref = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace_buffer, "NHD")
    wrapper_ref.plan(
        kv_indptr, kv_indices, kv_last_page_len,
        num_qo_heads, num_kv_heads, head_dim, page_size,
        window_left=window_left,
    )
    o_ref = wrapper_ref.run(q, (k_data, v_data))

    # Router: all ones = all SWA
    wrapper_router = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer, "NHD", use_router=True,
    )
    wrapper_router.plan(
        kv_indptr, kv_indices, kv_last_page_len,
        num_qo_heads, num_kv_heads, head_dim, page_size,
        window_left=window_left,
    )
    o_router = wrapper_router.run(q, (k_data, v_data), router=router)

    torch.testing.assert_close(o_router.cpu(), o_ref.cpu(), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("kv_len", [199, 1999])
@pytest.mark.parametrize("window_left", [33, 533])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("num_qo_heads", [4])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("page_size", [16])
def test_batch_decode_router_mixed(
    batch_size, kv_len, window_left, num_kv_heads, num_qo_heads, head_dim, page_size,
):
    """Mixed router: per-head outputs should match respective baselines (full or SWA)."""
    q = torch.randn(batch_size, num_qo_heads, head_dim, dtype=torch.float16, device="cuda:0")
    k_data, v_data, kv_indptr, kv_indices, kv_last_page_len = _make_paged_kv(
        batch_size, kv_len, num_kv_heads, head_dim, page_size,
    )
    # Mixed: first half of kv heads use SWA, rest use full attention
    router = torch.zeros(batch_size, num_kv_heads, dtype=torch.uint8, device="cuda:0")
    router[:, : num_kv_heads // 2] = 1  # first half = SWA

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device="cuda:0")

    # Full attention reference
    wrapper_full = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace_buffer, "NHD")
    wrapper_full.plan(
        kv_indptr, kv_indices, kv_last_page_len,
        num_qo_heads, num_kv_heads, head_dim, page_size,
    )
    o_full = wrapper_full.run(q, (k_data, v_data))

    # SWA reference
    wrapper_swa = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace_buffer, "NHD")
    wrapper_swa.plan(
        kv_indptr, kv_indices, kv_last_page_len,
        num_qo_heads, num_kv_heads, head_dim, page_size,
        window_left=window_left,
    )
    o_swa = wrapper_swa.run(q, (k_data, v_data))

    # Router attention
    wrapper_router = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer, "NHD", use_router=True,
    )
    wrapper_router.plan(
        kv_indptr, kv_indices, kv_last_page_len,
        num_qo_heads, num_kv_heads, head_dim, page_size,
        window_left=window_left,
    )
    o_router = wrapper_router.run(q, (k_data, v_data), router=router)

    # With MHA (num_qo_heads == num_kv_heads), each qo head maps to its kv head directly
    group_size = num_qo_heads // num_kv_heads
    for b in range(batch_size):
        for kv_h in range(num_kv_heads):
            qo_start = kv_h * group_size
            qo_end = qo_start + group_size
            if router[b, kv_h] == 1:  # SWA head
                torch.testing.assert_close(
                    o_router[b, qo_start:qo_end].cpu(),
                    o_swa[b, qo_start:qo_end].cpu(),
                    rtol=1e-3, atol=1e-3,
                )
            else:  # full attention head
                torch.testing.assert_close(
                    o_router[b, qo_start:qo_end].cpu(),
                    o_full[b, qo_start:qo_end].cpu(),
                    rtol=1e-3, atol=1e-3,
                )


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("kv_len", [199, 1999])
@pytest.mark.parametrize("window_left", [33, 533])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("num_qo_heads", [4])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("page_size", [16])
def test_batch_prefill_router_all_zeros(
    batch_size, kv_len, window_left, num_kv_heads, num_qo_heads, head_dim, page_size,
):
    """All-zeros router (all full attention) should match standard full attention for prefill."""
    qo_len = 17  # prefill query length per request
    q = torch.randn(batch_size * qo_len, num_qo_heads, head_dim, dtype=torch.float16, device="cuda:0")
    k_data, v_data, kv_indptr, kv_indices, kv_last_page_len = _make_paged_kv(
        batch_size, kv_len, num_kv_heads, head_dim, page_size,
    )
    qo_indptr = torch.arange(0, batch_size + 1, device="cuda:0", dtype=torch.int32) * qo_len
    router = torch.zeros(batch_size, num_kv_heads, dtype=torch.uint8, device="cuda:0")

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device="cuda:0")

    # Reference: standard full attention
    wrapper_ref = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace_buffer, "NHD")
    wrapper_ref.plan(
        qo_indptr, kv_indptr, kv_indices, kv_last_page_len,
        num_qo_heads, num_kv_heads, head_dim, page_size,
    )
    o_ref = wrapper_ref.run(q, (k_data, v_data))

    # Router: all zeros = all full attention
    wrapper_router = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer, "NHD", use_router=True,
    )
    wrapper_router.plan(
        qo_indptr, kv_indptr, kv_indices, kv_last_page_len,
        num_qo_heads, num_kv_heads, head_dim, page_size,
        window_left=window_left,
    )
    o_router = wrapper_router.run(q, (k_data, v_data), router=router)

    torch.testing.assert_close(o_router.cpu(), o_ref.cpu(), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("kv_len", [199, 1999])
@pytest.mark.parametrize("window_left", [33, 533])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("num_qo_heads", [4])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("page_size", [16])
def test_batch_prefill_router_all_ones(
    batch_size, kv_len, window_left, num_kv_heads, num_qo_heads, head_dim, page_size,
):
    """All-ones router (all SWA) should match standard SWA for prefill."""
    qo_len = 17
    q = torch.randn(batch_size * qo_len, num_qo_heads, head_dim, dtype=torch.float16, device="cuda:0")
    k_data, v_data, kv_indptr, kv_indices, kv_last_page_len = _make_paged_kv(
        batch_size, kv_len, num_kv_heads, head_dim, page_size,
    )
    qo_indptr = torch.arange(0, batch_size + 1, device="cuda:0", dtype=torch.int32) * qo_len
    router = torch.ones(batch_size, num_kv_heads, dtype=torch.uint8, device="cuda:0")

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device="cuda:0")

    # Reference: standard SWA
    wrapper_ref = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace_buffer, "NHD")
    wrapper_ref.plan(
        qo_indptr, kv_indptr, kv_indices, kv_last_page_len,
        num_qo_heads, num_kv_heads, head_dim, page_size,
        window_left=window_left,
    )
    o_ref = wrapper_ref.run(q, (k_data, v_data))

    # Router: all ones = all SWA
    wrapper_router = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer, "NHD", use_router=True,
    )
    wrapper_router.plan(
        qo_indptr, kv_indptr, kv_indices, kv_last_page_len,
        num_qo_heads, num_kv_heads, head_dim, page_size,
        window_left=window_left,
    )
    o_router = wrapper_router.run(q, (k_data, v_data), router=router)

    torch.testing.assert_close(o_router.cpu(), o_ref.cpu(), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("kv_len", [199, 1999])
@pytest.mark.parametrize("window_left", [33, 533])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("num_qo_heads", [4])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("page_size", [16])
def test_batch_prefill_router_mixed(
    batch_size, kv_len, window_left, num_kv_heads, num_qo_heads, head_dim, page_size,
):
    """Mixed router: per-head outputs should match respective baselines (full or SWA) for prefill."""
    qo_len = 17
    q = torch.randn(batch_size * qo_len, num_qo_heads, head_dim, dtype=torch.float16, device="cuda:0")
    k_data, v_data, kv_indptr, kv_indices, kv_last_page_len = _make_paged_kv(
        batch_size, kv_len, num_kv_heads, head_dim, page_size,
    )
    qo_indptr = torch.arange(0, batch_size + 1, device="cuda:0", dtype=torch.int32) * qo_len
    # Mixed: first half of kv heads use SWA, rest use full attention
    router = torch.zeros(batch_size, num_kv_heads, dtype=torch.uint8, device="cuda:0")
    router[:, : num_kv_heads // 2] = 1  # first half = SWA

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device="cuda:0")

    # Full attention reference
    wrapper_full = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace_buffer, "NHD")
    wrapper_full.plan(
        qo_indptr, kv_indptr, kv_indices, kv_last_page_len,
        num_qo_heads, num_kv_heads, head_dim, page_size,
    )
    o_full = wrapper_full.run(q, (k_data, v_data))

    # SWA reference
    wrapper_swa = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace_buffer, "NHD")
    wrapper_swa.plan(
        qo_indptr, kv_indptr, kv_indices, kv_last_page_len,
        num_qo_heads, num_kv_heads, head_dim, page_size,
        window_left=window_left,
    )
    o_swa = wrapper_swa.run(q, (k_data, v_data))

    # Router attention
    wrapper_router = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer, "NHD", use_router=True,
    )
    wrapper_router.plan(
        qo_indptr, kv_indptr, kv_indices, kv_last_page_len,
        num_qo_heads, num_kv_heads, head_dim, page_size,
        window_left=window_left,
    )
    o_router = wrapper_router.run(q, (k_data, v_data), router=router)

    # With MHA (num_qo_heads == num_kv_heads), each qo head maps to its kv head directly
    group_size = num_qo_heads // num_kv_heads
    for b in range(batch_size):
        for kv_h in range(num_kv_heads):
            qo_start = kv_h * group_size
            qo_end = qo_start + group_size
            # o shape for prefill is [batch_size * qo_len, num_qo_heads, head_dim]
            row_start = b * qo_len
            row_end = row_start + qo_len
            if router[b, kv_h] == 1:  # SWA head
                torch.testing.assert_close(
                    o_router[row_start:row_end, qo_start:qo_end].cpu(),
                    o_swa[row_start:row_end, qo_start:qo_end].cpu(),
                    rtol=1e-3, atol=1e-3,
                )
            else:  # full attention head
                torch.testing.assert_close(
                    o_router[row_start:row_end, qo_start:qo_end].cpu(),
                    o_full[row_start:row_end, qo_start:qo_end].cpu(),
                    rtol=1e-3, atol=1e-3,
                )
