"""
Microbenchmark to verify tile-level KV-skip in per-head router decode kernel.

Kernel under test: include/flashinfer/attention/decode.cuh, lines 420-616.
Design claim: when router[batch, kv_head] == 1, out-of-window tiles are skipped
  via cp_async::pred_load (K: @p predicate, V: src_size=0), saving DRAM reads.

We measure four configurations at each (seq_len, window) point:
  1. plain_full   : no router, window_left=-1           (baseline: full attn)
  2. plain_swa    : no router, window_left=W            (baseline: native SWA)
  3. router_all_full : use_router=True, router=zeros    (all heads = full)
  4. router_all_swa  : use_router=True, router=ones     (all heads = SWA)

Expected relationships if KV-skip works:
  - (4) time scales with W/seq_len, approaching (2)
  - (4) ≪ (3); (3) ≈ (1)
If KV-skip is a no-op:
  - (4) ≈ (3) ≈ (1) (router only gates compute, not memory)

Usage:
  python benchmarks/bench_router_kv_skip.py
  python benchmarks/bench_router_kv_skip.py --seq-len 16384 --batch 16
  python benchmarks/bench_router_kv_skip.py --use-tensor-cores   # test prefill kernel
"""

import argparse
import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch

import flashinfer
from flashinfer.testing.utils import bench_gpu_time


@dataclass
class Config:
    batch_size: int
    seq_len: int
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    page_size: int
    q_dtype: torch.dtype
    kv_dtype: torch.dtype
    use_tensor_cores: bool


def build_paged_kv(cfg: Config, device: str = "cuda:0"):
    seq_lens = torch.full((cfg.batch_size,), cfg.seq_len, dtype=torch.int32)
    seq_lens_blocks = torch.ceil(seq_lens / cfg.page_size).int()
    kv_indptr = torch.cat(
        [torch.tensor([0], dtype=torch.int32), torch.cumsum(seq_lens_blocks, 0).int()]
    ).to(device)
    last_page_len = (seq_lens - (seq_lens_blocks - 1) * cfg.page_size).int().to(device)
    num_blocks = int(kv_indptr[-1].item())
    kv_indices = torch.arange(num_blocks, dtype=torch.int32, device=device)

    q = torch.randn(
        cfg.batch_size, cfg.num_qo_heads, cfg.head_dim, dtype=cfg.q_dtype, device=device
    )
    # NHD: [num_blocks, 2, page_size, num_kv_heads, head_dim]
    kv_data = torch.randn(
        num_blocks,
        2,
        cfg.page_size,
        cfg.num_kv_heads,
        cfg.head_dim,
        device=device,
    ).to(cfg.kv_dtype)
    k_cache = kv_data[:, 0]
    v_cache = kv_data[:, 1]
    return q, k_cache, v_cache, kv_indptr, kv_indices, last_page_len


def make_wrapper(cfg: Config, workspace: torch.Tensor, use_router: bool):
    return flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace,
        kv_layout="NHD",
        use_tensor_cores=cfg.use_tensor_cores,
        use_router=use_router,
    )


def time_run(wrapper, q, k_cache, v_cache, router: Optional[torch.Tensor], n_warmup: int = 3):
    for _ in range(n_warmup):
        if router is not None:
            wrapper.run(q, (k_cache, v_cache), router=router)
        else:
            wrapper.run(q, (k_cache, v_cache))
    torch.cuda.synchronize()

    if router is not None:
        fn = lambda: wrapper.run(q, (k_cache, v_cache), router=router)
    else:
        fn = lambda: wrapper.run(q, (k_cache, v_cache))

    times = bench_gpu_time(fn, enable_cupti=True, cold_l2_cache=True)
    return float(np.median(times))


def bench_point(cfg: Config, window_left: int, workspace: torch.Tensor, swa_fraction: float):
    q, k_cache, v_cache, kv_indptr, kv_indices, last_page_len = build_paged_kv(cfg)

    plan_kwargs = dict(
        indptr=kv_indptr,
        indices=kv_indices,
        last_page_len=last_page_len,
        num_qo_heads=cfg.num_qo_heads,
        num_kv_heads=cfg.num_kv_heads,
        head_dim=cfg.head_dim,
        page_size=cfg.page_size,
        data_type=cfg.kv_dtype,
        q_data_type=cfg.q_dtype,
    )

    # (1) plain_full: no router, no SWA
    w_pf = make_wrapper(cfg, workspace, use_router=False)
    w_pf.plan(**plan_kwargs, window_left=-1)
    t_pf = time_run(w_pf, q, k_cache, v_cache, router=None)

    # (2) plain_swa: no router, window_left=W
    w_ps = make_wrapper(cfg, workspace, use_router=False)
    w_ps.plan(**plan_kwargs, window_left=window_left)
    t_ps = time_run(w_ps, q, k_cache, v_cache, router=None)

    # (3) router_all_full: router=zeros (all heads treated as full inside kernel)
    w_rf = make_wrapper(cfg, workspace, use_router=True)
    w_rf.plan(**plan_kwargs, window_left=window_left)
    router_zero = torch.zeros(
        cfg.batch_size, cfg.num_kv_heads, dtype=torch.uint8, device="cuda:0"
    )
    t_rf = time_run(w_rf, q, k_cache, v_cache, router=router_zero)

    # (4) router_all_swa: router=ones (all heads take SWA skip path)
    w_rs = make_wrapper(cfg, workspace, use_router=True)
    w_rs.plan(**plan_kwargs, window_left=window_left)
    router_one = torch.ones(
        cfg.batch_size, cfg.num_kv_heads, dtype=torch.uint8, device="cuda:0"
    )
    t_rs = time_run(w_rs, q, k_cache, v_cache, router=router_one)

    # (5) router_mixed: <swa_fraction> of (batch, head) pairs are SWA, rest full.
    # Deterministic pattern: mark the first N pairs (flat index) as SWA.
    w_rm = make_wrapper(cfg, workspace, use_router=True)
    w_rm.plan(**plan_kwargs, window_left=window_left)
    n_pairs = cfg.batch_size * cfg.num_kv_heads
    n_swa = int(round(n_pairs * swa_fraction))
    router_mixed_flat = torch.zeros(n_pairs, dtype=torch.uint8, device="cuda:0")
    # Spread the SWA pairs evenly — pick every k-th pair — so each KV head slot
    # has roughly the same SWA count. Avoids clumping all SWA into first batches.
    perm = torch.arange(n_pairs, device="cuda:0")
    swa_indices = perm[torch.linspace(0, n_pairs - 1, n_swa, device="cuda:0").long()]
    router_mixed_flat[swa_indices] = 1
    router_mixed = router_mixed_flat.view(cfg.batch_size, cfg.num_kv_heads)
    actual_swa_fraction = router_mixed.float().mean().item()
    t_rm = time_run(w_rm, q, k_cache, v_cache, router=router_mixed)

    return {
        "plain_full": t_pf,
        "plain_swa": t_ps,
        "router_all_full": t_rf,
        "router_all_swa": t_rs,
        "router_mixed": t_rm,
        "actual_swa_fraction": actual_swa_fraction,
    }


def pretty_print(cfg: Config, windows: List[int], results: List[dict], swa_fraction: float):
    print()
    print("=" * 110)
    print(
        f"batch={cfg.batch_size} seq_len={cfg.seq_len} qo_heads={cfg.num_qo_heads} "
        f"kv_heads={cfg.num_kv_heads} head_dim={cfg.head_dim} page={cfg.page_size} "
        f"kv_dtype={cfg.kv_dtype} tensor_cores={cfg.use_tensor_cores} "
        f"mixed_swa_frac={results[0]['actual_swa_fraction']:.3f}"
    )
    print("=" * 110)
    hdr = (
        f"{'window':>8} | {'plain_full':>11} {'router_full':>12} {'router_swa':>11} "
        f"{'router_mix':>11} | {'mix/full':>9} {'rswa/full':>10} {'mix_ideal':>10} {'swa_ideal':>10}"
    )
    print(hdr)
    print("-" * len(hdr))
    for W, r in zip(windows, results):
        swa_ideal = min(W, cfg.seq_len) / cfg.seq_len
        f_swa = r["actual_swa_fraction"]
        # Linear model: mixed time ≈ (1-f)*full_kv + f*swa_kv + fixed
        # Approximated as: mix_ideal = (1-f)*1.0 + f*swa_ideal
        mix_ideal = (1.0 - f_swa) * 1.0 + f_swa * swa_ideal
        ratio_mix = r["router_mixed"] / r["plain_full"]
        ratio_savings = r["router_all_swa"] / r["plain_full"]
        print(
            f"{W:>8} | "
            f"{r['plain_full']*1000:>10.3f}us "
            f"{r['router_all_full']*1000:>11.3f}us "
            f"{r['router_all_swa']*1000:>10.3f}us "
            f"{r['router_mixed']*1000:>10.3f}us | "
            f"{ratio_mix:>9.3f} "
            f"{ratio_savings:>10.3f} "
            f"{mix_ideal:>10.3f} "
            f"{swa_ideal:>10.3f}"
        )
    print()
    print("Interpretation:")
    print(f"  router_mix   = router_all_swa with {swa_fraction:.0%} of (batch, kv_head) pairs as SWA")
    print("  mix/full     = router_mixed time / plain_full time")
    print("  mix_ideal    = (1-f)*1.0 + f*(W/N)  [linear BW-prop model]")
    print("                 If mix/full ~= mix_ideal, the mix scales as expected.")
    print("                 If mix/full >> mix_ideal, the 10% full-heads are dominating.")
    print("                 If mix/full ~= 1.0 regardless, the slowest head serializes the batch.")
    print("  rswa/full    = router_all_swa / plain_full (pure SWA case, for comparison)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=8192)
    ap.add_argument("--num-qo-heads", type=int, default=32)
    ap.add_argument("--num-kv-heads", type=int, default=4)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--page-size", type=int, default=16)
    ap.add_argument("--kv-dtype", choices=["bf16", "fp16", "fp8"], default="bf16")
    ap.add_argument("--use-tensor-cores", action="store_true",
                    help="Use prefill kernel path instead of cuda-core decode kernel")
    ap.add_argument("--windows", type=int, nargs="+",
                    default=None,
                    help="Window sizes to sweep (default: 128, 512, 1024, 2048, seq_len)")
    ap.add_argument("--swa-fraction", type=float, default=0.9,
                    help="Fraction of (batch, kv_head) pairs set to SWA in the mixed-router variant")
    args = ap.parse_args()

    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp8": torch.float8_e4m3fn,
    }
    q_dtype = torch.bfloat16 if args.kv_dtype == "fp8" else dtype_map[args.kv_dtype]
    kv_dtype = dtype_map[args.kv_dtype]

    cfg = Config(
        batch_size=args.batch,
        seq_len=args.seq_len,
        num_qo_heads=args.num_qo_heads,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        page_size=args.page_size,
        q_dtype=q_dtype,
        kv_dtype=kv_dtype,
        use_tensor_cores=args.use_tensor_cores,
    )

    windows = args.windows
    if windows is None:
        windows = [128, 512, 1024, 2048, cfg.seq_len]
    windows = [w for w in windows if w <= cfg.seq_len]

    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device="cuda:0")

    results = [bench_point(cfg, W, workspace, args.swa_fraction) for W in windows]
    pretty_print(cfg, windows, results, args.swa_fraction)


if __name__ == "__main__":
    main()
