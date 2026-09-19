"""Pair up the vLLM A/B runs and report the paired difference.

    python bench/analyze_ab.py

`integration/run_ab.sh` writes alternating runs into two files, one with each
backend going first. This pairs adjacent runs and reports the median paired
difference.

*Pairs, not pooled means*: each pair ran back to back on the same GPU state, so
the within-pair difference cancels drift.

*Both orderings*: the second backend in a pair sees a warmer GPU, so equal
numbers of pairs each way cancel that bias.

The result is printed twice, with and without runs slower than 2x the global
median -- this machine has stalls that swallow whole runs, and dropping an
outlier silently is how a benchmark starts lying. If the two disagree, the
measurement did not resolve the difference.
"""

from __future__ import annotations

import json
import pathlib
import statistics as st
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _pairs(path: pathlib.Path, first: str) -> list[tuple[float, float]]:
    if not path.exists():
        return []
    runs = json.loads(path.read_text())["runs"]
    out = []
    for i in range(0, len(runs) - 1, 2):
        a, b = runs[i], runs[i + 1]
        if a["backend"] != first:
            print(f"  ! {path.name} pair {i//2}: expected {first} first, got "
                  f"{a['backend']} -- skipped", file=sys.stderr)
            continue
        d = {a["backend"]: a["tpot_ms_p50"], b["backend"]: b["tpot_ms_p50"]}
        out.append((d["vllm-default"], d["ours"]))
    return out


def main() -> None:
    fwd = _pairs(ROOT / "results/vllm_e2e.json", "vllm-default")
    rev = _pairs(ROOT / "results/vllm_e2e_rev.json", "ours")
    allp = fwd + rev
    if not allp:
        sys.exit("no paired runs found; run integration/run_ab.sh first")

    print(f"{len(fwd)} pairs default-first, {len(rev)} pairs ours-first\n")
    print(f"{'order':>9s} {'default':>9s} {'ours':>9s} {'delta':>8s}")
    for tag, group in (("forward", fwd), ("reverse", rev)):
        for d, o in group:
            print(f"{tag:>9s} {d:9.2f} {o:9.2f} {100*(o-d)/d:+7.1f}%")

    def report(pairs: list[tuple[float, float]], label: str) -> None:
        deltas = [100 * (o - d) / d for d, o in pairs]
        ds = [d for d, _ in pairs]
        os_ = [o for _, o in pairs]
        print(f"\n{label} ({len(pairs)} pairs)")
        print(f"  vLLM default : median {st.median(ds):6.2f} ms  "
              f"[{min(ds):.1f} - {max(ds):.1f}]")
        print(f"  ours         : median {st.median(os_):6.2f} ms  "
              f"[{min(os_):.1f} - {max(os_):.1f}]")
        print(f"  paired delta : median {st.median(deltas):+6.1f} %  "
              f"[{min(deltas):+.1f} - {max(deltas):+.1f}]")

    report(allp, "all runs")
    med = st.median([v for p in allp for v in p])
    kept = [(d, o) for d, o in allp if d < 2 * med and o < 2 * med]
    if len(kept) < len(allp):
        report(kept, f"excluding {len(allp)-len(kept)} pair(s) with a run >2x median")
        print("\n  Both numbers are shown on purpose. If they agree, the outliers "
              "were\n  system noise; if they disagreed, the run would not have "
              "resolved anything.")


if __name__ == "__main__":
    main()
