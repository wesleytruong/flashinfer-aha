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

from functools import partial
from pathlib import Path

import pytest
import torch

import flashinfer
from flashinfer.jit import env as jit_env


def _wire_editable_checkout() -> None:
    root = Path(__file__).resolve().parents[2]
    if (root / "csrc").exists() and (root / "include").exists():
        jit_env.FLASHINFER_CSRC_DIR = root / "csrc"
        jit_env.FLASHINFER_INCLUDE_DIR = root / "include"
        jit_env.CUTLASS_INCLUDE_DIRS = [
            root / "3rdparty/cutlass/include",
            root / "3rdparty/cutlass/tools/util/include",
        ]
        jit_env.SPDLOG_INCLUDE_DIR = root / "3rdparty/spdlog/include"


_wire_editable_checkout()


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


def _decode_sink_recent_reference(q, k_data, v_data, router, kv_len, window_left, sink_size):
    """Dense reference for per-KV-head full vs sink+recent decode."""
    batch_size, num_qo_heads, head_dim = q.shape
    num_kv_heads = router.shape[1]
    group_size = num_qo_heads // num_kv_heads
    pages_per_seq = k_data.shape[0] // batch_size
    page_size = k_data.shape[1]
    k_dense = k_data.view(
        batch_size, pages_per_seq, page_size, num_kv_heads, head_dim
    ).reshape(batch_size, pages_per_seq * page_size, num_kv_heads, head_dim)[:, :kv_len]
    v_dense = v_data.view(
        batch_size, pages_per_seq, page_size, num_kv_heads, head_dim
    ).reshape(batch_size, pages_per_seq * page_size, num_kv_heads, head_dim)[:, :kv_len]

    out = torch.empty_like(q)
    pos = torch.arange(kv_len, device=q.device)
    recent_start = max(0, kv_len - window_left - 1)
    for b in range(batch_size):
        for kv_h in range(num_kv_heads):
            if router[b, kv_h] == 0:
                mask = torch.ones(kv_len, dtype=torch.bool, device=q.device)
            else:
                mask = (pos < sink_size) | (pos >= recent_start)
            k_sel = k_dense[b, mask, kv_h].float()
            v_sel = v_dense[b, mask, kv_h].float()
            for qo_h in range(kv_h * group_size, (kv_h + 1) * group_size):
                scores = (q[b, qo_h].float() @ k_sel.T) * (head_dim ** -0.5)
                probs = torch.softmax(scores, dim=-1)
                out[b, qo_h] = (probs @ v_sel).to(out.dtype)
    return out


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


def test_batch_decode_router_sink_recent_matches_dense_reference():
    """router_sink_size turns router-local heads into AHA/Duo sink+recent heads."""
    torch.manual_seed(7)
    batch_size = 2
    kv_len = 97
    window_left = 17
    sink_size = 11
    num_kv_heads = 2
    num_qo_heads = 4
    head_dim = 64
    page_size = 16

    q = torch.randn(batch_size, num_qo_heads, head_dim, dtype=torch.float16, device="cuda:0")
    k_data, v_data, kv_indptr, kv_indices, kv_last_page_len = _make_paged_kv(
        batch_size, kv_len, num_kv_heads, head_dim, page_size,
    )
    router = torch.tensor([[1, 0], [0, 1]], dtype=torch.uint8, device="cuda:0")
    expected = _decode_sink_recent_reference(
        q, k_data, v_data, router, kv_len, window_left, sink_size,
    )

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device="cuda:0")
    wrapper_router = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer, "NHD", use_router=True, use_aha_router=True,
    )
    wrapper_router.plan(
        kv_indptr, kv_indices, kv_last_page_len,
        num_qo_heads, num_kv_heads, head_dim, page_size,
        window_left=window_left,
    )
    # AHA-native API: 1 = full, 0 = sink+recent local. This is the inverse
    # of the legacy router tensor used by the original FlashInfer fork tests.
    aha_gate = (~router.bool()).to(torch.uint8)
    actual = wrapper_router.run(
        q, (k_data, v_data), aha_gate=aha_gate, router_sink_size=sink_size,
    )

    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=2e-3, atol=2e-3)


def test_tensor_core_fast_decode_aha_all_full_uses_full_plan():
    """AHA fast_decode_plan must not shrink full heads to window_left."""
    torch.manual_seed(17)
    batch_size = 2
    kv_len = 4096
    window_left = 255
    num_kv_heads = 4
    num_qo_heads = 8
    head_dim = 128
    page_size = 16

    q = torch.randn(
        batch_size,
        num_qo_heads,
        head_dim,
        dtype=torch.float16,
        device="cuda:0",
    )
    k_data, v_data, kv_indptr, kv_indices, kv_last_page_len = _make_paged_kv(
        batch_size, kv_len, num_kv_heads, head_dim, page_size,
    )
    workspace_buffer = torch.empty(
        128 * 1024 * 1024, dtype=torch.int8, device="cuda:0"
    )

    wrapper_ref = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer, "NHD", use_tensor_cores=True,
    )
    wrapper_ref.plan(
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        data_type=torch.float16,
        q_data_type=torch.float16,
    )
    expected = wrapper_ref.run(q, (k_data, v_data))

    wrapper_router = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer, "NHD", use_tensor_cores=True,
        use_router=True, use_aha_router=True,
    )
    wrapper_router.plan(
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        window_left=window_left,
        data_type=torch.float16,
        q_data_type=torch.float16,
    )
    wrapper_router.plan = partial(flashinfer.fast_decode_plan, wrapper_router)
    wrapper_router.plan(
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        window_left=window_left,
        data_type=torch.float16,
        q_data_type=torch.float16,
    )

    aha_gate = torch.ones(
        batch_size, num_kv_heads, dtype=torch.uint8, device="cuda:0"
    )
    actual = wrapper_router.run(
        q, (k_data, v_data), aha_gate=aha_gate, router_sink_size=0,
    )

    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=1e-3, atol=1e-3)


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


def _prefill_aha_sink_recent_reference(
    q, k_data, v_data, aha_gate, batch_size, qo_len, kv_len, window_left, sink_size
):
    """Dense reference for AHA per-token gates in paged prefill."""
    _, num_qo_heads, head_dim = q.shape
    num_kv_heads = aha_gate.shape[1]
    group_size = num_qo_heads // num_kv_heads
    pages_per_seq = k_data.shape[0] // batch_size
    page_size = k_data.shape[1]
    k_dense = k_data.view(
        batch_size, pages_per_seq, page_size, num_kv_heads, head_dim
    ).reshape(batch_size, pages_per_seq * page_size, num_kv_heads, head_dim)[:, :kv_len]
    v_dense = v_data.view(
        batch_size, pages_per_seq, page_size, num_kv_heads, head_dim
    ).reshape(batch_size, pages_per_seq * page_size, num_kv_heads, head_dim)[:, :kv_len]
    gate = aha_gate.view(batch_size, qo_len, num_kv_heads)
    q_view = q.view(batch_size, qo_len, num_qo_heads, head_dim)
    out = torch.empty_like(q_view)
    pos = torch.arange(kv_len, device=q.device)

    for b in range(batch_size):
        for qi in range(qo_len):
            q_abs = kv_len - qo_len + qi
            causal = pos <= q_abs
            recent = (q_abs - pos) <= window_left
            sink = pos < sink_size
            for kv_h in range(num_kv_heads):
                if gate[b, qi, kv_h] != 0:
                    mask = causal
                else:
                    mask = causal & (sink | recent)
                k_sel = k_dense[b, mask, kv_h].float()
                v_sel = v_dense[b, mask, kv_h].float()
                for qo_h in range(kv_h * group_size, (kv_h + 1) * group_size):
                    scores = (q_view[b, qi, qo_h].float() @ k_sel.T) * (head_dim ** -0.5)
                    probs = torch.softmax(scores, dim=-1)
                    out[b, qi, qo_h] = (probs @ v_sel).to(out.dtype)
    return out.reshape(batch_size * qo_len, num_qo_heads, head_dim)


@pytest.mark.parametrize("gate_mode", ["all0", "all1", "random"])
def test_batch_prefill_aha_token_router_sink_recent_matches_dense_reference(gate_mode):
    """AHA prefill uses per-token gates: 1 = full, 0 = sink+recent local."""
    torch.manual_seed(11)
    batch_size = 1
    qo_len = 128
    kv_len = 128
    window_left = 31
    sink_size = 8
    num_kv_heads = 8
    num_qo_heads = 16
    head_dim = 128
    page_size = 16

    q = torch.randn(
        batch_size * qo_len, num_qo_heads, head_dim,
        dtype=torch.float16, device="cuda:0",
    )
    k_data, v_data, kv_indptr, kv_indices, kv_last_page_len = _make_paged_kv(
        batch_size, kv_len, num_kv_heads, head_dim, page_size,
    )
    qo_indptr = torch.arange(
        0, batch_size + 1, device="cuda:0", dtype=torch.int32
    ) * qo_len
    if gate_mode == "all0":
        aha_gate = torch.zeros(
            batch_size * qo_len, num_kv_heads, dtype=torch.uint8, device="cuda:0",
        )
    elif gate_mode == "all1":
        aha_gate = torch.ones(
            batch_size * qo_len, num_kv_heads, dtype=torch.uint8, device="cuda:0",
        )
    else:
        aha_gate = (
            torch.rand(batch_size * qo_len, num_kv_heads, device="cuda:0") < 0.5
        ).to(torch.uint8)

    expected = _prefill_aha_sink_recent_reference(
        q, k_data, v_data, aha_gate, batch_size, qo_len, kv_len, window_left, sink_size,
    )

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device="cuda:0")
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer, "NHD", use_router=True, use_aha_router=True, backend="fa2",
    )
    wrapper.plan(
        qo_indptr, kv_indptr, kv_indices, kv_last_page_len,
        num_qo_heads, num_kv_heads, head_dim, page_size,
        causal=True, window_left=window_left,
    )
    actual = wrapper.run(
        q, (k_data, v_data), router=aha_gate,
        router_sink_size=sink_size, router_is_aha_gate=True,
    )

    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=2e-3, atol=2e-2)
