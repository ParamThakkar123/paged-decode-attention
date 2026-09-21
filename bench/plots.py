"""Figures for the real-model run.

    python -m bench.plots --outdir docs

Two figures, both from data produced by a real Qwen2.5-0.5B run through vLLM:

  decode_kernel_breakdown.png  where GPU time actually goes in a decode step,
                               aggregated from the torch.profiler chrome trace
  vllm_ab_tpot.png             the 12 paired end-to-end runs behind the
                               parity claim, with the machine's outliers shown

The breakdown reads the chrome trace rather than the printed profiler table,
because that table interleaves ATen wrapper ops (`aten::mm`) with the leaf CUDA
kernels they launch. Summing those columns double counts. Trace events with
`cat == "kernel"` are unambiguously the GPU kernels and nothing else.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re
import statistics as st
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Colors: one accent for our kernel, muted greys for everything else, so the
# figure answers "how much is ours" before it answers anything else.
OURS = "#c2410c"
GEMM = "#475569"
ATTN_OTHER = "#0f766e"
MISC = "#94a3b8"


def _classify(name: str) -> tuple[str, str]:
    """(bucket, color) for a CUDA kernel name."""
    if name.startswith("_paged_decode_kernel") or "split_reduce" in name:
        return "Attention (our kernel)", OURS
    if "unified_attention" in name or "flash" in name.lower():
        return "Attention (vLLM, prefill)", ATTN_OTHER
    if re.search(r"gemm|cutlass|cublas|xmma|splitKreduce", name, re.I):
        return "GEMM (weights)", GEMM
    if name.startswith("triton_") or "elementwise" in name or "reduce_kernel" in name:
        return "Norm / activation / elementwise", MISC
    return "Other", MISC


def kernel_breakdown(trace: pathlib.Path, out: pathlib.Path) -> dict:
    raw = json.loads(trace.read_text(encoding="utf-8"))
    events = raw["traceEvents"] if isinstance(raw, dict) else raw
    kernels = [e for e in events if e.get("cat") == "kernel"]
    if not kernels:
        raise SystemExit(f"no kernel events in {trace}")

    per_name: collections.Counter[str] = collections.Counter()
    per_count: collections.Counter[str] = collections.Counter()
    for e in kernels:
        per_name[e["name"]] += e.get("dur", 0.0)
        per_count[e["name"]] += 1
    total_us = sum(per_name.values())

    buckets: collections.Counter[str] = collections.Counter()
    for name, us in per_name.items():
        buckets[_classify(name)[0]] += us

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13, 5.6), gridspec_kw={"width_ratios": [1.15, 1]})

    # ---- left: time by category -------------------------------------------
    order = sorted(buckets.items(), key=lambda kv: kv[1], reverse=True)
    labels = [k for k, _ in order]
    vals = [v / 1000.0 for _, v in order]
    colors = [OURS if "our kernel" in lb
              else (ATTN_OTHER if lb.startswith("Attention")
                    else (GEMM if lb.startswith("GEMM") else MISC))
              for lb in labels]
    bars = ax1.barh(range(len(labels)), vals, color=colors)
    ax1.set_yticks(range(len(labels)))
    ax1.set_yticklabels(labels, fontsize=9)
    ax1.invert_yaxis()
    ax1.set_xlabel("GPU kernel time (ms)")
    ax1.set_title("Where decode-step GPU time goes", fontsize=11, loc="left")
    for b, v in zip(bars, vals):
        ax1.text(b.get_width() + total_us / 1000 * 0.012,
                 b.get_y() + b.get_height() / 2,
                 f"{v:,.0f} ms  ({100*v*1000/total_us:.1f} %)",
                 va="center", fontsize=8.5)
    ax1.set_xlim(0, max(vals) * 1.32)
    ax1.spines[["top", "right"]].set_visible(False)

    # ---- right: the individual kernels ------------------------------------
    top = per_name.most_common(9)
    names, tvals = [], []
    for n, us in top:
        short = re.sub(r"^void ", "", n).split("<")[0][:38]
        names.append(f"{short}  x{per_count[n]:,}")
        tvals.append(us / 1000.0)
    tcolors = [_classify(n)[1] for n, _ in top]
    ax2.barh(range(len(names)), tvals, color=tcolors)
    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels(names, fontsize=7.5)
    ax2.invert_yaxis()
    ax2.set_xlabel("GPU kernel time (ms)")
    ax2.set_title("Top kernels by total device time", fontsize=11, loc="left")
    ax2.spines[["top", "right"]].set_visible(False)

    ours_us = buckets.get("Attention (our kernel)", 0.0)
    fig.suptitle(
        "Qwen2.5-0.5B-Instruct through vLLM 0.9.2, RTX 3050 - torch.profiler (CUPTI), "
        f"{len(kernels):,} kernel launches\n"
        f"our decode kernel: {ours_us/1000:,.0f} ms of {total_us/1000:,.0f} ms "
        f"= {100*ours_us/total_us:.1f} % of GPU time, "
        f"{per_count['_paged_decode_kernel']:,} launches "
        f"@ {per_name['_paged_decode_kernel']/max(per_count['_paged_decode_kernel'],1):.1f} us",
        fontsize=9.5, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")
    return {"total_us": total_us, "ours_us": ours_us,
            "ours_pct": 100 * ours_us / total_us,
            "launches": per_count["_paged_decode_kernel"]}


def _pairs(path: pathlib.Path, first: str) -> list[tuple[float, float]]:
    if not path.exists():
        return []
    runs = json.loads(path.read_text(encoding="utf-8"))["runs"]
    out = []
    for i in range(0, len(runs) - 1, 2):
        a, b = runs[i], runs[i + 1]
        if a["backend"] != first:
            continue
        d = {a["backend"]: a["tpot_ms_p50"], b["backend"]: b["tpot_ms_p50"]}
        out.append((d["vllm-default"], d["ours"]))
    return out


def ab_tpot(out: pathlib.Path) -> dict:
    fwd = _pairs(ROOT / "results/vllm_e2e.json", "vllm-default")
    rev = _pairs(ROOT / "results/vllm_e2e_rev.json", "ours")
    allp = fwd + rev
    if not allp:
        raise SystemExit("no paired runs in results/")

    med_all = st.median([v for p in allp for v in p])
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12, 5.2), gridspec_kw={"width_ratios": [1.5, 1]})

    # ---- left: every pair, as a slope -------------------------------------
    inl = [(d, o) for d, o in allp if d < 2 * med_all and o < 2 * med_all]
    hi = max(max(d, o) for d, o in inl) * 1.10
    lo = min(min(d, o) for d, o in inl) * 0.92
    n_off = 0
    for d, o in allp:
        if d > 2 * med_all or o > 2 * med_all:
            n_off += 1
            continue
        ax1.plot([0, 1], [d, o], "-o", ms=5, lw=1.5,
                 color=OURS if o < d else GEMM, alpha=0.85, zorder=2)
    ax1.plot([0, 1], [st.median([d for d, _ in inl]), st.median([o for _, o in inl])],
             "-", lw=3.5, color="#0f172a", zorder=3, label="median")
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(["vLLM's attention", "our kernel"])
    ax1.set_xlim(-0.25, 1.25)
    ax1.set_ylim(lo, hi)
    ax1.set_ylabel("TPOT (ms/token)")
    ax1.set_title(f"{len(allp)} paired runs, both orderings", fontsize=11, loc="left")
    ax1.legend(fontsize=8, loc="upper right", framealpha=0.95)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.grid(axis="y", alpha=0.25)
    if n_off:
        ax1.text(0.5, 0.02,
                 f"{n_off} further pairs off-scale (vLLM at 91 and 97 ms): "
                 "machine stalls, shown in grey at right",
                 transform=ax1.transAxes, ha="center", fontsize=7.5,
                 color="#64748b")

    # ---- right: the paired differences ------------------------------------
    marked = sorted(((100 * (o - d) / d, d > 2 * med_all or o > 2 * med_all)
                     for d, o in allp), key=lambda t: t[0])
    deltas = [x for x, _ in marked]
    kept = [100 * (o - d) / d for d, o in allp
            if d < 2 * med_all and o < 2 * med_all]
    cols = ["#cbd5e1" if bad else (OURS if x < 0 else GEMM) for x, bad in marked]
    ax2.barh(range(len(deltas)), deltas, color=cols)
    ax2.axvline(0, color="#0f172a", lw=1)
    ax2.axvline(st.median(kept), color="#c2410c", ls="--", lw=1.4,
                label=f"median (outliers excl.) {st.median(kept):+.1f} %")
    ax2.axvspan(-10, 10, color="#e2e8f0", alpha=0.6, zorder=0,
                label="approx. noise floor (+/-10 %)")
    ax2.set_xlabel("TPOT change, ours vs vLLM (%)   negative = ours faster")
    ax2.set_yticks([])
    ax2.set_title("Per-pair difference", fontsize=11, loc="left")
    ax2.legend(fontsize=8, loc="lower right", framealpha=0.95)
    ax2.spines[["top", "right", "left"]].set_visible(False)

    fig.suptitle(
        "End-to-end TPOT: our attention backend vs vLLM's own\n"
        "Qwen2.5-0.5B-Instruct, 24 requests x 96 output tokens, RTX 3050 - "
        "the two grey pairs are machine stalls, not the backend",
        fontsize=10, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")
    return {"pairs": len(allp), "median_delta_pct": st.median(kept)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="docs")
    ap.add_argument("--trace", default="results/profile/decode-ours.json")
    args = ap.parse_args()

    outdir = ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    trace = ROOT / args.trace
    if trace.exists():
        kernel_breakdown(trace, outdir / "decode_kernel_breakdown.png")
    else:
        print(f"skipping kernel breakdown: {trace} not present "
              "(run bench_vllm.py --profile first)", file=sys.stderr)

    ab_tpot(outdir / "vllm_ab_tpot.png")


if __name__ == "__main__":
    main()
