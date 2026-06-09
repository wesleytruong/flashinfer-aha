#!/usr/bin/env python3
"""Context-length sweep for the per-head router KV-skip decode kernel.

Companion to bench_router_kv_skip.py. Fixes batch / window / heads and sweeps the
context length (seq_len), so you can see how the router's pruning speedup grows as
the full-attention baseline scales with context while windowed heads stay ~flat.

Configs per point (same family as bench_router_kv_skip.py):
  plain_full   : no router, full attention            (baseline)
  router_full  : use_router=True, router=zeros         (all heads global, f=0)
  router_swa   : use_router=True, router=ones          (all heads local,  f=1)
  mix@<f>      : use_router=True, <f> of (batch, kv_head) pairs local
                (one column per --mixed-fractions value; default 50/70/90%)

Usage:
  python benchmarks/bench_router_context_sweep.py
  python benchmarks/bench_router_context_sweep.py --batch 8 --window 128 \
      --contexts 2048 8192 32768 131072 --mixed-fractions 0.5 0.7 0.9
"""
import argparse

import numpy as np
import torch

from bench_router_kv_skip import Config, build_paged_kv, make_wrapper, time_run


def spread_router(batch, n_kv, frac, device="cuda:0"):
    """Deterministic even spread of `frac` SWA pairs over (batch, kv_head)."""
    n = batch * n_kv
    r = torch.zeros(n, dtype=torch.uint8, device=device)
    k = int(round(n * frac))
    if k > 0:
        idx = torch.linspace(0, n - 1, k, device=device).long()
        r[idx] = 1
    return r.view(batch, n_kv)


def measure(cfg, window, fractions, ws):
    """Return (plain_full, router_full, router_swa, [mix per fraction]) in µs."""
    q, k_cache, v_cache, indptr, indices, last = build_paged_kv(cfg)
    pk = dict(
        indptr=indptr, indices=indices, last_page_len=last,
        num_qo_heads=cfg.num_qo_heads, num_kv_heads=cfg.num_kv_heads,
        head_dim=cfg.head_dim, page_size=cfg.page_size,
        data_type=cfg.kv_dtype, q_data_type=cfg.q_dtype,
    )
    wpf = make_wrapper(cfg, ws, use_router=False)
    wpf.plan(**pk, window_left=-1)
    t_pf = time_run(wpf, q, k_cache, v_cache, router=None)

    def run_router(router):
        w = make_wrapper(cfg, ws, use_router=True)
        w.plan(**pk, window_left=window)
        return time_run(w, q, k_cache, v_cache, router=router)

    r0 = torch.zeros(cfg.batch_size, cfg.num_kv_heads, dtype=torch.uint8, device="cuda:0")
    t_rf = run_router(r0)
    t_rs = run_router(torch.ones_like(r0))
    mixes = [run_router(spread_router(cfg.batch_size, cfg.num_kv_heads, f)) for f in fractions]

    del q, k_cache, v_cache, indptr, indices, last
    torch.cuda.empty_cache()
    return (t_pf * 1000, t_rf * 1000, t_rs * 1000, [m * 1000 for m in mixes])


def print_tables(axis_name, rows, header_desc, fractions):
    # Column order: all-global, mix (low->high local fraction), all-local.
    order = sorted(range(len(fractions)), key=lambda i: fractions[i])
    fr = [f"mix@{int(round(fractions[i] * 100))}%" for i in order]
    print()
    print("=" * 104)
    print(header_desc)
    print("=" * 104)

    print("\nAbsolute kernel time (µs, lower = better)")
    h = (f"{axis_name:>9} | {'plain_full':>11} | {'all-global':>11} "
         + " ".join(f"{c:>10}" for c in fr) + f" {'all-local':>11}")
    print(h)
    print("-" * len(h))
    for ax, pf, rf, rs, mixes in rows:
        ms = " ".join(f"{mixes[i]:>9.1f}u" for i in order)
        print(f"{ax:>9} | {pf:>10.1f}u | {rf:>10.1f}u {ms} {rs:>10.1f}u")

    print("\nSpeedup vs plain_full (higher = better)")
    h2 = (f"{axis_name:>9} | {'all-global':>11} "
          + " ".join(f"{c:>9}" for c in fr) + f" {'all-local':>10}")
    print(h2)
    print("-" * len(h2))
    for ax, pf, rf, rs, mixes in rows:
        ms = " ".join(f"{pf / mixes[i]:>8.2f}x" for i in order)
        print(f"{ax:>9} | {pf / rf:>10.2f}x {ms} {pf / rs:>9.2f}x")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--window", type=int, default=128)
    ap.add_argument("--num-qo-heads", type=int, default=16)
    ap.add_argument("--num-kv-heads", type=int, default=16)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--page-size", type=int, default=16)
    ap.add_argument("--kv-dtype", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--cuda-core", action="store_true",
                    help="Use the cuda-core decode kernel (default: tensor-core 3D split-KV path)")
    ap.add_argument("--mixed-fractions", type=float, nargs="+", default=[0.5, 0.7, 0.9],
                    help="Local-head fractions for the mixed-router columns")
    ap.add_argument("--contexts", type=int, nargs="+",
                    default=[2048, 4096, 8192, 16384, 32768, 65536, 131072])
    args = ap.parse_args()

    dt = torch.bfloat16 if args.kv_dtype == "bf16" else torch.float16
    ws = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device="cuda:0")

    rows = []
    for ctx in args.contexts:
        cfg = Config(
            batch_size=args.batch, seq_len=ctx, num_qo_heads=args.num_qo_heads,
            num_kv_heads=args.num_kv_heads, head_dim=args.head_dim,
            page_size=args.page_size, q_dtype=dt, kv_dtype=dt,
            use_tensor_cores=not args.cuda_core,
        )
        try:
            row = measure(cfg, args.window, args.mixed_fractions, ws)
        except torch.cuda.OutOfMemoryError:
            print(f"  seq_len={ctx}: OOM, skipping")
            torch.cuda.empty_cache()
            continue
        rows.append((ctx, *row))

    path = "CUDA-core" if args.cuda_core else "tensor-core"
    desc = (f"CONTEXT SWEEP  |  batch={args.batch}  window={args.window}  "
            f"heads={args.num_qo_heads}qo/{args.num_kv_heads}kv  head_dim={args.head_dim}  "
            f"{args.kv_dtype}  {path} decode")
    print_tables("seq_len", rows, desc, args.mixed_fractions)


if __name__ == "__main__":
    main()
