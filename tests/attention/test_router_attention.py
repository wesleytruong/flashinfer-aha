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
@pytest.mark.parametrize("use_tensor_cores", [False, True])
def test_batch_decode_router_all_zeros(
    batch_size, kv_len, window_left, num_kv_heads, num_qo_heads, head_dim, page_size,
    use_tensor_cores,
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
        workspace_buffer, "NHD", use_router=True, use_tensor_cores=use_tensor_cores,
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
@pytest.mark.parametrize("use_tensor_cores", [False, True])
def test_batch_decode_router_all_ones(
    batch_size, kv_len, window_left, num_kv_heads, num_qo_heads, head_dim, page_size,
    use_tensor_cores,
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
        workspace_buffer, "NHD", use_router=True, use_tensor_cores=use_tensor_cores,
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
@pytest.mark.parametrize("use_tensor_cores", [False, True])
def test_batch_decode_router_mixed(
    batch_size, kv_len, window_left, num_kv_heads, num_qo_heads, head_dim, page_size,
    use_tensor_cores,
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
        workspace_buffer, "NHD", use_router=True, use_tensor_cores=use_tensor_cores,
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


def test_batch_decode_router_cudagraph_per_step():
    """Regression for the vLLM AHA windowing bug.

    Under CUDA graph, run() captures the router *pointer* and reads live contents at
    that frozen address. The vLLM bug was allocating a fresh router tensor every step
    (gate_hard.to(uint8)), so the captured kernel kept reading the stale capture-time
    buffer -> per-step routing ignored, every replay reused the captured pattern
    (flat timing, global == local). The fix writes the router IN PLACE into a
    persistent buffer the graph captured.

    This test captures a graph that reads a persistent router buffer, then flips the
    buffer in place across "steps" and asserts the output follows the *current* router:
    router==0 -> full attention, router==1 -> SWA, mixed -> per-head. It also flips
    back to global to prove the kernel is not frozen on the captured pattern.
    """
    from flashinfer.utils import get_compute_capability

    if get_compute_capability(torch.device("cuda:0"))[0] < 8:
        pytest.skip("tensor-core decode path requires SM80+")

    batch_size, kv_len = 2, 2048
    num_heads, head_dim, page_size, window = 16, 128, 16, 128
    device = "cuda:0"

    q = torch.randn(batch_size, num_heads, head_dim, dtype=torch.float16, device=device)
    k_data, v_data, kv_indptr, kv_indices, kv_last_page_len = _make_paged_kv(
        batch_size, kv_len, num_heads, head_dim, page_size,
    )
    ws = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)

    # References (non-router, eager): full attention and SWA.
    ref_full = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, "NHD")
    ref_full.plan(kv_indptr, kv_indices, kv_last_page_len,
                  num_heads, num_heads, head_dim, page_size)
    o_full = ref_full.run(q, (k_data, v_data)).cpu()

    ref_swa = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, "NHD")
    ref_swa.plan(kv_indptr, kv_indices, kv_last_page_len,
                 num_heads, num_heads, head_dim, page_size, window_left=window)
    o_swa = ref_swa.run(q, (k_data, v_data)).cpu()

    # The test is only meaningful if full != SWA (kv_len must exceed the window).
    assert not torch.allclose(o_full, o_swa, rtol=1e-2, atol=1e-2)

    # Router wrapper: tensor-core + cudagraph + persistent buffers (incl. router).
    router_buf = torch.zeros(batch_size, num_heads, dtype=torch.uint8, device=device)
    w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        ws, "NHD", use_cuda_graph=True, use_tensor_cores=True, use_router=True,
        paged_kv_indptr_buffer=kv_indptr,
        paged_kv_indices_buffer=kv_indices,
        paged_kv_last_page_len_buffer=kv_last_page_len,
    )
    w.plan(kv_indptr, kv_indices, kv_last_page_len,
           num_heads, num_heads, head_dim, page_size, window_left=window)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            out = w.run(q, (k_data, v_data), router=router_buf)
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = w.run(q, (k_data, v_data), router=router_buf)

    def replay_with(pattern):
        router_buf.copy_(pattern)  # in-place into the captured buffer
        g.replay()
        torch.cuda.synchronize()
        return out.cpu().clone()

    # router==0 everywhere -> full attention
    o_global = replay_with(torch.zeros_like(router_buf))
    torch.testing.assert_close(o_global, o_full, rtol=1e-2, atol=1e-2)

    # router==1 everywhere -> SWA  (also pins polarity: 1 == SWA)
    o_local = replay_with(torch.ones_like(router_buf))
    torch.testing.assert_close(o_local, o_swa, rtol=1e-2, atol=1e-2)

    # mixed: even heads SWA, odd heads full -> per-head routing under cudagraph
    mixed = torch.zeros_like(router_buf)
    mixed[:, ::2] = 1
    o_mixed = replay_with(mixed)
    for b in range(batch_size):
        for h in range(num_heads):
            expected = o_swa[b, h] if mixed[b, h] == 1 else o_full[b, h]
            torch.testing.assert_close(o_mixed[b, h], expected, rtol=1e-2, atol=1e-2)

    # flip back to global: proves replay is not frozen on the captured pattern
    o_global_again = replay_with(torch.zeros_like(router_buf))
    torch.testing.assert_close(o_global_again, o_full, rtol=1e-2, atol=1e-2)
