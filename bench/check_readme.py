"""Re-derive every headline number in the README from the result JSON.

    python bench/check_readme.py

A README is a cache of numbers that live somewhere else, and caches go stale.
This one went stale twice during development -- a test count that stopped
matching the suite, and a vLLM TPOT that survived a change to how the two
backends were launched. Both read as perfectly credible.

So the claims are checked mechanically instead: each entry below recomputes a
number from its source file and compares it to what the README says, and the
last few assert that specific superseded strings are *absent*. It exits non-zero
on any mismatch, which is the only way this stays true after the next re-run.
"""

from __future__ import annotations

import json
import pathlib
import statistics as st
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
R = lambda f: json.loads((ROOT / f).read_text())  # noqa: E731


def main() -> int:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    checks: list[tuple[bool, str, str]] = []

    def chk(name: str, ok: bool, detail: str = "") -> None:
        checks.append((bool(ok), name, detail))

    # -- section 4.1: achieved bandwidth -----------------------------------
    tri = [r for r in R("results/sweep_fp16.json")["rows"]
           if r.get("impl") == "triton" and r.get("pct_peak")]
    good = [r["pct_peak"] for r in tri if r["batch"] >= 4 and r["seqlen"] >= 2048]
    chk("98-103 % at batch>=4, ctx>=2k", 97.5 <= min(good) and max(good) <= 103.5,
        f"{min(good):.1f}-{max(good):.1f}")
    b4 = [r["pct_peak"] for r in tri if r["batch"] >= 4]
    chk("94 % worst case at batch>=4", abs(min(b4) - 94.4) < 0.6, f"{min(b4):.1f}")
    chk("462 sweep measurements", len(R("results/sweep_fp16.json")["rows"]) == 462)

    # -- section 6: the decode loop ----------------------------------------
    sv = {r["impl"]: r for r in R("results/serving.json")["results"]}
    chk("Triton TPOT p50 0.773 ms", abs(sv["triton"]["tpot_ms_p50"] - 0.773) < 0.005,
        f"{sv['triton']['tpot_ms_p50']:.3f}")
    chk("Triton 42,120 tok/s", abs(sv["triton"]["tokens_per_s"] - 42120) < 60)
    chk("CUDA 40,083 tok/s", abs(sv["cuda"]["tokens_per_s"] - 40083) < 60)
    chk("SDPA 283.6 ms p50", abs(sv["sdpa_memeff"]["tpot_ms_p50"] - 283.6) < 0.5)
    chk("KV utilization 13.5 %", abs(sv["triton"]["kv_utilization"] * 100 - 13.5) < 0.6)

    # -- section 4.6: shape generality -------------------------------------
    sh = R("results/shapes.json")
    tp = [r["triton"]["pct_peak"] for r in sh["rows"]]
    cu = [r["cuda"]["pct_peak"] for r in sh["rows"] if r.get("cuda")]
    chk("shapes: Triton 96-102 %", 95.0 <= min(tp) and max(tp) <= 103.0,
        f"{min(tp):.1f}-{max(tp):.1f}")
    chk("shapes: CUDA floor 61 %", 60 <= min(cu) <= 62, f"{min(cu):.1f}")
    chk("5 CUDA instantiations", len(sh["cuda_supported_shapes"]) == 5)
    chk("10 shape points", len(sh["rows"]) == 10)

    # -- section 6.1: the vLLM engine A/B ----------------------------------
    def pairs(f: str, first: str) -> list[tuple[float, float]]:
        p = ROOT / f
        if not p.exists():
            return []
        runs = json.loads(p.read_text())["runs"]
        out = []
        for i in range(0, len(runs) - 1, 2):
            a, b = runs[i], runs[i + 1]
            if a["backend"] != first:
                continue
            d = {a["backend"]: a["tpot_ms_p50"], b["backend"]: b["tpot_ms_p50"]}
            out.append((d["vllm-default"], d["ours"]))
        return out

    ap = pairs("results/vllm_e2e.json", "vllm-default") + \
        pairs("results/vllm_e2e_rev.json", "ours")
    if ap:
        chk("12 paired vLLM runs", len(ap) == 12, str(len(ap)))
        med = st.median([v for p in ap for v in p])
        kept = [(d, o) for d, o in ap if d < 2 * med and o < 2 * med]
        chk("trimmed medians 15.93 / 15.70",
            abs(st.median([d for d, _ in kept]) - 15.93) < 0.01
            and abs(st.median([o for _, o in kept]) - 15.70) < 0.01)
        chk("trimmed paired delta -1.8 %",
            abs(st.median([100 * (o - d) / d for d, o in kept]) + 1.8) < 0.1)
        chk("untrimmed paired delta -2.6 %",
            abs(st.median([100 * (o - d) / d for d, o in ap]) + 2.6) < 0.1)

    # -- claims that must still be true, and ones that must be gone --------
    import subprocess
    r = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q", "--co"],
                       capture_output=True, text=True, cwd=ROOT)
    n_tests = sum(1 for ln in r.stdout.splitlines() if "::" in ln)
    chk(f"README's test count matches the suite ({n_tests})",
        f"{n_tests} tests" in readme and f"{n_tests} correctness" in readme,
        f"suite has {n_tests}")

    for stale in ("61 tests", "61 correctness", "14.40 ms TPOT", "never executed",
                  "1,623 tok/s", "371 successful"):
        chk(f"superseded claim absent: {stale!r}", stale not in readme)

    for ok, name, detail in checks:
        print(("PASS  " if ok else "FAIL  ") + name + (f"   [{detail}]" if detail else ""))
    n_ok = sum(1 for o, _, _ in checks if o)
    print(f"\n{n_ok}/{len(checks)} consistent")
    return 0 if n_ok == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
