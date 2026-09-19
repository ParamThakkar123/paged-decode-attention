"""Figures for the README, generated from the sweep JSON.

    python -m bench.plots results/sweep_fp16.json --outdir docs

Produces:
  bandwidth_vs_context.png   achieved DRAM bandwidth (% of measured peak)
  latency_vs_batch.png       latency scaling, log-log
  launch_overhead.png        eager vs CUDA-graph, and the share that is host cost
  cuda_variants.png          the CUDA kernel's unroll/warp A/B
  quantization.png           fp16 vs fp8_e5m2 vs int8   (quant sweep only)

These are plots of the benchmark's own output rather than screenshots of the
Nsight GUI: they regenerate from the JSON, they diff, and they carry the axis
labels that make the claim checkable. The `.ncu-rep` / `.nsys-rep` files in
`results/` are there for the GUI views.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

STYLE = {
    "triton": ("#1f77b4", "-", "o"),
    "triton_nosplit": ("#1f77b4", "--", "s"),
    "triton_pertoken_bt": ("#1f77b4", ":", "^"),
    "cuda": ("#d62728", "-", "o"),
    "cuda:v1_naive": ("#d62728", "--", "s"),
    "cuda:v3_tuned": ("#d62728", ":", "^"),
    "cuda_nosplit": ("#ff7f0e", "--", "v"),
    "sdpa_math": ("#7f7f7f", "-", "x"),
    "sdpa_cudnn": ("#8c564b", "-", "P"),
    "sdpa_memeff": ("#2ca02c", "-", "d"),
    "sdpa_flash": ("#9467bd", "-", "*"),
}
LABEL = {
    "triton": "Triton (split-KV)",
    "triton_nosplit": "Triton, no split-KV",
    "triton_pertoken_bt": "Triton, per-token BT",
    "cuda": "CUDA v2 (unroll 4)",
    "cuda:v1_naive": "CUDA v1 (no unroll)",
    "cuda:v3_tuned": "CUDA v3 (8 warps)",
    "cuda_nosplit": "CUDA, no split-KV",
    "sdpa_math": "SDPA math",
    "sdpa_cudnn": "SDPA cuDNN (GQA)",
    "sdpa_memeff": "SDPA mem-efficient",
    "sdpa_flash": "SDPA flash (FA2)",
}


def _ok(rows, **f):
    return [r for r in rows if r["ok"] and all(r[k] == v for k, v in f.items())]


def _finish(fig, ax, path, title, xlabel, ylabel, legend=True):
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3, linewidth=0.6)
    if legend:
        ax.legend(fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  wrote {path}")


def bandwidth_vs_context(data, outdir: pathlib.Path) -> None:
    rows = data["rows"]
    batches = sorted({r["batch"] for r in rows if r["ok"]})
    picks = [b for b in (1, 8, 32) if b in batches] or batches[:3]
    fig, axes = plt.subplots(1, len(picks), figsize=(4.6 * len(picks), 3.8), sharey=True)
    axes = [axes] if len(picks) == 1 else list(axes)

    for ax, b in zip(axes, picks):
        for impl in STYLE:
            pts = sorted(_ok(rows, impl=impl, batch=b, kv_dtype="fp16"),
                         key=lambda r: r["seqlen"])
            if not pts:
                continue
            c, ls, mk = STYLE[impl]
            ax.plot([p["seqlen"] for p in pts], [p["pct_peak"] for p in pts],
                    color=c, linestyle=ls, marker=mk, markersize=4, linewidth=1.4,
                    label=LABEL[impl])
        ax.axhline(100, color="k", linewidth=0.8, linestyle="-.", alpha=0.6)
        ax.set_xscale("log", base=2)
        ax.set_ylim(0, 115)
        ax.set_title(f"batch = {b}", fontsize=10)
        ax.set_xlabel("context length (tokens)")
        ax.grid(alpha=0.3, linewidth=0.6)
    axes[0].set_ylabel("% of measured peak DRAM read BW")
    axes[-1].legend(fontsize=7, framealpha=0.9, loc="lower right")
    fig.suptitle(
        f"Decode attention: achieved bandwidth vs context  "
        f"({data['gpu']['name']}, peak {data['peak_read_gbs_measured']:.0f} GB/s)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(outdir / "bandwidth_vs_context.png", dpi=150)
    plt.close(fig)
    print(f"  wrote {outdir / 'bandwidth_vs_context.png'}")


def latency_vs_batch(data, outdir: pathlib.Path) -> None:
    rows = data["rows"]
    seqlens = sorted({r["seqlen"] for r in rows if r["ok"]})
    ctx = 4096 if 4096 in seqlens else seqlens[len(seqlens) // 2]
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for impl in STYLE:
        pts = sorted(_ok(rows, impl=impl, seqlen=ctx, kv_dtype="fp16"),
                     key=lambda r: r["batch"])
        if not pts:
            continue
        c, ls, mk = STYLE[impl]
        ax.plot([p["batch"] for p in pts], [p["ms"] * 1e3 for p in pts],
                color=c, linestyle=ls, marker=mk, markersize=4, linewidth=1.4,
                label=LABEL[impl])
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    _finish(fig, ax, outdir / "latency_vs_batch.png",
            f"Decode attention latency vs batch (context {ctx:,})",
            "batch size", "latency (us, log)")


def launch_overhead(data, outdir: pathlib.Path) -> None:
    rows = [r for r in data["rows"]
            if r["ok"] and r["impl"] == "triton" and r["kv_dtype"] == "fp16"
            and r.get("ms_graph")]
    if not rows:
        return
    seqlens = sorted({r["seqlen"] for r in rows})
    ctx = 4096 if 4096 in seqlens else seqlens[0]
    pts = sorted([r for r in rows if r["seqlen"] == ctx], key=lambda r: r["batch"])
    if not pts:
        return
    batches = [p["batch"] for p in pts]
    graph = [p["ms_graph"] * 1e3 for p in pts]
    over = [(p["ms_eager"] - p["ms_graph"]) * 1e3 for p in pts]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.0, 4.0))
    x = range(len(batches))
    ax.bar(x, graph, label="kernel (CUDA-graph replay)", color="#1f77b4")
    ax.bar(x, over, bottom=graph, label="host launch overhead", color="#d62728", alpha=0.75)
    ax.set_xticks(list(x))
    ax.set_xticklabels(batches)
    ax.set_xlabel("batch size")
    ax.set_ylabel("latency (us)")
    ax.set_title(f"Where a decode step goes (Triton, context {ctx:,})", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y", linewidth=0.6)

    share = [100 * o / (g + o) for g, o in zip(graph, over)]
    ax2.plot(batches, share, marker="o", color="#d62728", linewidth=1.6)
    ax2.set_xscale("log", base=2)
    ax2.set_ylim(0, max(100, max(share) * 1.15))
    ax2.set_xlabel("batch size")
    ax2.set_ylabel("host overhead, % of eager step")
    ax2.set_title("Launch overhead dominates at small batch", fontsize=10)
    ax2.grid(alpha=0.3, linewidth=0.6)

    fig.tight_layout()
    fig.savefig(outdir / "launch_overhead.png", dpi=150)
    plt.close(fig)
    print(f"  wrote {outdir / 'launch_overhead.png'}")


def cuda_variants(data, outdir: pathlib.Path) -> None:
    rows = data["rows"]
    impls = [i for i in ("cuda:v1_naive", "cuda", "cuda:v3_tuned", "triton")
             if _ok(rows, impl=i)]
    if len(impls) < 2:
        return
    seqlens = sorted({r["seqlen"] for r in rows if r["ok"]})
    ctx = 4096 if 4096 in seqlens else seqlens[len(seqlens) // 2]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for impl in impls:
        pts = sorted(_ok(rows, impl=impl, seqlen=ctx, kv_dtype="fp16"),
                     key=lambda r: r["batch"])
        if not pts:
            continue
        c, ls, mk = STYLE[impl]
        ax.plot([p["batch"] for p in pts], [p["pct_peak"] for p in pts],
                color=c, linestyle=ls, marker=mk, markersize=5, linewidth=1.5,
                label=LABEL[impl])
    ax.set_xscale("log", base=2)
    ax.axhline(100, color="k", linewidth=0.8, linestyle="-.", alpha=0.6)
    _finish(fig, ax, outdir / "cuda_variants.png",
            f"CUDA kernel tuning A/B vs Triton (context {ctx:,})",
            "batch size", "% of measured peak DRAM read BW")


def quantization(data, outdir: pathlib.Path) -> None:
    rows = data["rows"]
    dtypes = sorted({r["kv_dtype"] for r in rows if r["ok"]})
    if len(dtypes) < 2:
        return
    seqlens = sorted({r["seqlen"] for r in rows if r["ok"]})
    ctx = seqlens[-1]
    batches = sorted({r["batch"] for r in rows if r["ok"] and r["seqlen"] == ctx})
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    width = 0.8 / len(dtypes)
    colors = {"fp16": "#1f77b4", "fp8_e5m2": "#ff7f0e", "int8": "#2ca02c"}
    for i, dt in enumerate(dtypes):
        vals, xs = [], []
        for j, b in enumerate(batches):
            hit = _ok(rows, impl="triton", batch=b, seqlen=ctx, kv_dtype=dt)
            if hit:
                xs.append(j + i * width)
                vals.append(hit[0]["ms"] * 1e3)
        ax.bar(xs, vals, width=width, label=dt, color=colors.get(dt))
    ax.set_xticks([j + 0.4 - width / 2 for j in range(len(batches))])
    ax.set_xticklabels(batches)
    _finish(fig, ax, outdir / "quantization.png",
            f"KV-cache dtype: decode latency (Triton, context {ctx:,})",
            "batch size", "latency (us)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep", nargs="?", default="results/sweep_fp16.json")
    ap.add_argument("--outdir", default="docs")
    ap.add_argument("--only", default=None,
                    help="comma-separated subset of figures to regenerate. Needed "
                         "because the quantization sweep only covers a few batch "
                         "sizes; letting it redraw the others would silently "
                         "replace the full-sweep figures with sparser ones.")
    args = ap.parse_args()

    data = json.loads(pathlib.Path(args.sweep).read_text(encoding="utf-8"))
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    figures = {
        "bandwidth": bandwidth_vs_context,
        "latency": latency_vs_batch,
        "overhead": launch_overhead,
        "variants": cuda_variants,
        "quantization": quantization,
    }
    wanted = [x.strip() for x in args.only.split(",")] if args.only else list(figures)
    for name in wanted:
        if name not in figures:
            raise SystemExit(f"unknown figure {name!r}; choose from {list(figures)}")
        figures[name](data, outdir)


if __name__ == "__main__":
    main()
