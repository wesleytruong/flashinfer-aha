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
  python benchmarks/bench_router_kv_skip.py --cuda-core   # cuda-core decode (default is tensor-core 3D)
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
    # NHD: [num_blocks, 2, page_size, num_kv_heads, head_dim].
    # Generate directly in the target dtype where randn supports it (bf16/fp16) so we
    # don't transiently allocate an fp32 copy — at GH200 scale (1M ctx / batch 256) the
    # fp32->bf16 cast peak is ~3x the bf16 footprint and OOMs before the kernel runs.
    fp8 = cfg.kv_dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    gen_dtype = torch.bfloat16 if fp8 else cfg.kv_dtype
    kv_data = torch.randn(
        num_blocks,
        2,
        cfg.page_size,
        cfg.num_kv_heads,
        cfg.head_dim,
        dtype=gen_dtype,
        device=device,
    )
    if fp8:
        kv_data = kv_data.to(cfg.kv_dtype)
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


def spread_router(batch, n_kv, frac, device="cuda:0"):
    """Deterministic even spread of `frac` SWA pairs over (batch, kv_head)."""
    n = batch * n_kv
    r = torch.zeros(n, dtype=torch.uint8, device=device)
    k = int(round(n * frac))
    if k > 0:
        idx = torch.linspace(0, n - 1, k, device=device).long()
        r[idx] = 1
    return r.view(batch, n_kv)


def bench_point(cfg: Config, window_left: int, workspace: torch.Tensor, fractions):
    """Measure plain_full, router_full (f=0), router_swa (f=1), and mix at each fraction."""
    q, k_cache, v_cache, kv_indptr, kv_indices, last_page_len = build_paged_kv(cfg)

    plan_kwargs = dict(
        indptr=kv_indptr, indices=kv_indices, last_page_len=last_page_len,
        num_qo_heads=cfg.num_qo_heads, num_kv_heads=cfg.num_kv_heads,
        head_dim=cfg.head_dim, page_size=cfg.page_size,
        data_type=cfg.kv_dtype, q_data_type=cfg.q_dtype,
    )

    # plain_full: no router, full attention (baseline)
    w_pf = make_wrapper(cfg, workspace, use_router=False)
    w_pf.plan(**plan_kwargs, window_left=-1)
    t_pf = time_run(w_pf, q, k_cache, v_cache, router=None)

    def run_router(router):
        w = make_wrapper(cfg, workspace, use_router=True)
        w.plan(**plan_kwargs, window_left=window_left)
        return time_run(w, q, k_cache, v_cache, router=router)

    r0 = torch.zeros(cfg.batch_size, cfg.num_kv_heads, dtype=torch.uint8, device="cuda:0")
    t_rf = run_router(r0)                       # router_all_full (f=0)
    t_rs = run_router(torch.ones_like(r0))      # router_all_swa  (f=1)
    mixes = [run_router(spread_router(cfg.batch_size, cfg.num_kv_heads, f)) for f in fractions]

    return {
        "plain_full": t_pf, "router_all_full": t_rf, "router_all_swa": t_rs, "mixes": mixes,
    }


def pretty_print(cfg: Config, windows: List[int], results: List[dict], fractions):
    # Column order: all-global, mix (low->high local fraction), all-local.
    order = sorted(range(len(fractions)), key=lambda i: fractions[i])
    fr = [f"mix@{int(round(fractions[i] * 100))}%" for i in order]
    print()
    print("=" * 104)
    print(
        f"WINDOW SWEEP  |  batch={cfg.batch_size} seq_len={cfg.seq_len} "
        f"heads={cfg.num_qo_heads}qo/{cfg.num_kv_heads}kv head_dim={cfg.head_dim} "
        f"page={cfg.page_size} kv_dtype={cfg.kv_dtype} tensor_cores={cfg.use_tensor_cores}"
    )
    print("=" * 104)

    print("\nAbsolute kernel time (µs, lower = better)")
    h = (f"{'window':>8} | {'plain_full':>11} | {'all-global':>11} "
         + " ".join(f"{c:>10}" for c in fr) + f" {'all-local':>11}")
    print(h)
    print("-" * len(h))
    for W, r in zip(windows, results):
        ms = " ".join(f"{r['mixes'][i] * 1000:>9.1f}u" for i in order)
        print(f"{W:>8} | {r['plain_full']*1000:>10.1f}u | {r['router_all_full']*1000:>10.1f}u "
              f"{ms} {r['router_all_swa']*1000:>10.1f}u")

    print("\nSpeedup vs plain_full (higher = better)")
    h2 = (f"{'window':>8} | {'all-global':>11} "
          + " ".join(f"{c:>9}" for c in fr) + f" {'all-local':>10}")
    print(h2)
    print("-" * len(h2))
    for W, r in zip(windows, results):
        pf = r["plain_full"]
        ms = " ".join(f"{pf / r['mixes'][i]:>8.2f}x" for i in order)
        print(f"{W:>8} | {pf / r['router_all_full']:>10.2f}x {ms} {pf / r['router_all_swa']:>9.2f}x")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=32)  # GH200 (120GB): fill the GPU; 5090 default was 8
    ap.add_argument("--seq-len", type=int, default=128000)
    ap.add_argument("--num-qo-heads", type=int, default=16)
    ap.add_argument("--num-kv-heads", type=int, default=16)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--page-size", type=int, default=16)
    ap.add_argument("--kv-dtype", choices=["bf16", "fp16", "fp8"], default="bf16")
    ap.add_argument("--cuda-core", action="store_true",
                    help="Use the cuda-core decode kernel (default: tensor-core 3D split-KV path)")
    ap.add_argument("--windows", type=int, nargs="+",
                    default=None,
                    help="Window sizes to sweep "
                         "(default: 128, 512, 2048, 8192, 32768, 65536, 131072, seq_len; capped at seq_len)")
    ap.add_argument("--mixed-fractions", type=float, nargs="+", default=[0.5, 0.7, 0.9],
                    help="Local-head fractions for the mixed-router columns")
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
        use_tensor_cores=not args.cuda_core,
    )

    windows = args.windows
    if windows is None:
        # ×4 geometric sweep, plus seq_len as the full-attention reference.
        # Extended past 32768 for GH200 (120GB) — at batch=32/seq=128k the larger
        # windows still fit; entries above seq_len are dropped by the cap below.
        windows = [128, 512, 2048, 8192, 32768, 65536, 131072, cfg.seq_len]
    windows = sorted({w for w in windows if w <= cfg.seq_len})

    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device="cuda:0")

    results = [bench_point(cfg, W, workspace, args.mixed_fractions) for W in windows]
    pretty_print(cfg, windows, results, args.mixed_fractions)


if __name__ == "__main__":
    main()
