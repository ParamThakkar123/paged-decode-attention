"""Turn a sweep JSON into the Markdown tables the README quotes.

    python -m bench.report results/sweep.json
    python -m bench.report results/sweep.json --out results/TABLES.md
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import defaultdict

PRETTY = {
    "triton": "Triton (split-KV, per-page BT)",
    "triton_nosplit": "Triton, no split-KV",
    "triton_pertoken_bt": "Triton, per-token block table",
    "cuda": "CUDA v2 (4 warps, unroll 4)",
    "cuda:v1_naive": "CUDA v1 (4 warps, no unroll)",
    "cuda:v3_tuned": "CUDA v3 (8 warps, unroll 4)",
    "cuda_nosplit": "CUDA, no split-KV",
    "sdpa_math": "PyTorch SDPA (math)",
    "sdpa_cudnn": "PyTorch SDPA (cuDNN, native GQA)",
    "sdpa_memeff": "PyTorch SDPA (mem-efficient)",
    "sdpa_flash": "PyTorch SDPA (flash / FA2)",
}


def _ok_rows(rows, **filt):
    for r in rows:
        if not r["ok"]:
            continue
        if all(r[k] == v for k, v in filt.items()):
            yield r


def main_table(data, kv_dtype="fp16") -> str:
    rows = data["rows"]
    peak = data["peak_read_gbs_measured"]
    impls = [i for i in PRETTY if any(r["impl"] == i for r in rows)]
    pts = sorted({(r["batch"], r["seqlen"]) for r in rows if r["kv_dtype"] == kv_dtype})

    out = [
        f"Latency in microseconds (median), KV dtype `{kv_dtype}`. "
        f"`--` = did not run (out of VRAM budget or unsupported shape).",
        "",
        "| batch | ctx | " + " | ".join(PRETTY[i] for i in impls) + " |",
        "|---:|---:|" + "---:|" * len(impls),
    ]
    for b, s in pts:
        cells = []
        for impl in impls:
            hit = next(_ok_rows(rows, impl=impl, batch=b, seqlen=s, kv_dtype=kv_dtype), None)
            cells.append(f"{hit['ms']*1e3:,.0f}" if hit else "--")
        out.append(f"| {b} | {s:,} | " + " | ".join(cells) + " |")

    out += [
        "",
        f"Achieved DRAM read bandwidth as a percentage of the measured peak "
        f"(~{peak:.0f} GB/s streaming read, re-measured once per context block so "
        f"each row is normalized against the ceiling the GPU had at that moment).",
        "",
        "| batch | ctx | " + " | ".join(PRETTY[i] for i in impls) + " |",
        "|---:|---:|" + "---:|" * len(impls),
    ]
    for b, s in pts:
        cells = []
        for impl in impls:
            hit = next(_ok_rows(rows, impl=impl, batch=b, seqlen=s, kv_dtype=kv_dtype), None)
            cells.append(f"{hit['pct_peak']:.0f}%" if hit else "--")
        out.append(f"| {b} | {s:,} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def speedup_table(data, target="triton", kv_dtype="fp16") -> str:
    """Speedup over both PyTorch baselines, side by side.

    `sdpa_memeff` is the one to quote: it is a real fused attention kernel and
    the comparison is apples to apples. `sdpa_math` is included for completeness
    but it degrades catastrophically once its materialized attention matrix and
    4x-expanded KV stop fitting comfortably -- the >100x entries are that
    thrashing, not a kernel-quality result, and should not be cited as a win.
    """
    rows = data["rows"]
    pts = sorted({(r["batch"], r["seqlen"]) for r in rows if r["kv_dtype"] == kv_dtype})
    out = [
        f"Speedup of `{PRETTY.get(target, target)}` over the PyTorch baselines, "
        "and its absolute decode throughput.",
        "",
        "| batch | ctx | vs SDPA mem-efficient | vs SDPA math | tokens/s (ours) |",
        "|---:|---:|---:|---:|---:|",
    ]
    any_row = False
    for b, s in pts:
        t = next(_ok_rows(rows, impl=target, batch=b, seqlen=s, kv_dtype=kv_dtype), None)
        if not t:
            continue
        any_row = True
        cells = []
        for base in ("sdpa_memeff", "sdpa_math"):
            z = next(_ok_rows(rows, impl=base, batch=b, seqlen=s, kv_dtype=kv_dtype), None)
            if not z:
                cells.append("--")
            else:
                ratio = z["ms"] / t["ms"]
                # Flag baselines that are thrashing, not merely losing.
                cells.append(f"{ratio:.1f}x" + (" ⚠" if ratio > 100 else ""))
        out.append(f"| {b} | {s:,} | {cells[0]} | {cells[1]} | {t['tok_per_s']:,.0f} |")
    out.append("")
    out.append("⚠ = the baseline is thrashing under memory pressure at this point; "
               "the ratio is not a meaningful kernel comparison.")
    return "\n".join(out) if any_row else ""


def quant_table(data) -> str:
    rows = data["rows"]
    dtypes = sorted({r["kv_dtype"] for r in rows if r["ok"]})
    if len(dtypes) < 2:
        return ""
    pts = sorted({(r["batch"], r["seqlen"]) for r in rows if r["ok"]})
    out = [
        "KV-cache quantization, `triton` implementation. "
        "`GB/s` is the achieved read bandwidth over the *stored* bytes, so a "
        "1-byte cache moving the same tokens in half the time shows the same "
        "GB/s at half the latency.",
        "",
        "| batch | ctx | " + " | ".join(f"{d} us" for d in dtypes) + " | int8 vs fp16 |",
        "|---:|---:|" + "---:|" * (len(dtypes) + 1),
    ]
    for b, s in pts:
        cells, byname = [], {}
        for d in dtypes:
            hit = next(_ok_rows(rows, impl="triton", batch=b, seqlen=s, kv_dtype=d), None)
            byname[d] = hit
            cells.append(f"{hit['ms']*1e3:,.0f}" if hit else "--")
        if not any(byname.values()):
            continue
        sp = "--"
        if byname.get("fp16") and byname.get("int8"):
            sp = f"{byname['fp16']['ms']/byname['int8']['ms']:.2f}x"
        out.append(f"| {b} | {s:,} | " + " | ".join(cells) + f" | {sp} |")
    return "\n".join(out)


def overhead_table(data, impl="triton", kv_dtype="fp16") -> str:
    """Eager vs CUDA-graph replay: how much of a decode step is host overhead."""
    rows = [r for r in data["rows"]
            if r["ok"] and r["impl"] == impl and r["kv_dtype"] == kv_dtype
            and r.get("ms_graph")]
    if not rows:
        return ""
    out = [
        f"Host-side launch overhead for `{PRETTY.get(impl, impl)}`: the same call "
        "measured eagerly and under CUDA-graph replay. The gap is pure host cost "
        "and is what a serving runtime removes by capturing the decode step.",
        "",
        "| batch | ctx | eager us | graphed us | overhead us | overhead share |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(rows, key=lambda r: (r["batch"], r["seqlen"])):
        e, g = r["ms_eager"] * 1e3, r["ms_graph"] * 1e3
        out.append(f"| {r['batch']} | {r['seqlen']:,} | {e:,.0f} | {g:,.0f} | "
                   f"{e-g:,.0f} | {100*(e-g)/e:.0f}% |")
    return "\n".join(out)


def skipped_table(data) -> str:
    notes = defaultdict(list)
    for r in data["rows"]:
        if not r["ok"] and r["note"]:
            notes[r["note"].split(":")[0]].append(r)
    if not notes:
        return ""
    out = ["What did not run, and why.", "", "| reason | count | example |", "|---|---:|---|"]
    for reason, rs in sorted(notes.items(), key=lambda kv: -len(kv[1])):
        e = rs[0]
        out.append(f"| {reason} | {len(rs)} | `{e['impl']}` b={e['batch']} "
                   f"ctx={e['seqlen']} {e['kv_dtype']} |")
    return "\n".join(out)


def l2_warnings(data) -> str:
    bad = [r for r in data["rows"]
           if r["ok"] and (r.get("l2_resident_frac") or 0) > 0.5]
    if not bad:
        return ""
    pts = sorted({(r["batch"], r["seqlen"]) for r in bad})
    return ("> **Cache caveat.** At these points more than half the KV working set fits "
            "in this GPU's L2, so the measured bandwidth is partly L2 bandwidth, not "
            "DRAM: " + ", ".join(f"b={b}/ctx={s}" for b, s in pts) + ".")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep", nargs="?", default="results/sweep.json")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data = json.loads(pathlib.Path(args.sweep).read_text(encoding="utf-8"))
    gpu = data["gpu"]

    parts = [
        f"# Decode-attention sweep results",
        "",
        f"- GPU: **{gpu['name']}** (sm_{gpu['capability'].replace('.','')}, "
        f"{gpu['sm_count']} SMs, {gpu['total_vram_gb']} GiB, driver {gpu['driver_cuda']})",
        f"- PyTorch {gpu['torch']} / CUDA {gpu['torch_cuda']}",
        f"- Measured streaming-read peak: **{data['peak_read_gbs_measured']:.0f} GB/s**"
        + (f" (theoretical {data['peak_gbs_theoretical']:.0f} GB/s)"
           if data.get("peak_gbs_theoretical") else ""),
        f"- Page size {data['block_size']}, block tables **{data['block_table']}**",
        f"- Sweep wall time: {data['elapsed_s']} s",
        "",
    ]
    for kv in sorted({r["kv_dtype"] for r in data["rows"]}):
        if any(r["ok"] and r["kv_dtype"] == kv for r in data["rows"]):
            parts += [f"## {kv}", "", main_table(data, kv), ""]
    for tbl, title in (
        (speedup_table(data), "## Speedup vs the PyTorch baselines"),
        (overhead_table(data), "## Launch overhead (eager vs CUDA graph)"),
        (quant_table(data), "## KV-cache quantization"),
        (skipped_table(data), "## Coverage"),
        (l2_warnings(data), ""),
    ):
        if tbl:
            parts += ([title, ""] if title else []) + [tbl, ""]

    text = "\n".join(parts)
    if args.out:
        pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
