"""End-to-end vLLM benchmark: TPOT, throughput, and throughput at a latency SLO.

    # inside WSL2, in the cu126 environment (see integration/README.md):
    cd /mnt/e/Projects/inference_benchmark
    ~/vllm126/bin/python integration/bench_vllm.py --model <hf-id>

    # to compare our attention backend against vLLM's default:
    ~/vllm126/bin/python integration/bench_vllm.py --model <hf-id> --backend PAGEDATTN_TRITON

Unlike `bench/bench_serving.py`, which times the attention layer inside a
simulated decode loop, this runs a real model through the real engine. It is the
number that says whether the kernel is worth anything in a server, as opposed to
in a microbenchmark.

TPOT is measured one of two ways, and the result records which:

  * from vLLM's own per-request metrics, when the build populates them:
    `(last_token_time - first_token_time) / (output_tokens - 1)`
  * otherwise -- vLLM's **V1 engine does not populate `RequestOutput.metrics`** --
    by a two-point difference: run the same trace generating 1 token, then N,
    and take `(wall(N) - wall(1)) / (N - 1)`. Prefill, scheduling and
    tokenization appear in both terms and cancel. This yields one aggregate
    number rather than a distribution, so p90/p99 are suppressed rather than
    fabricated from a single sample.

Sizing note for a 4 GB card: the model weights, the CUDA context (~300 MiB) and
the KV pool all share the card. `--gpu-memory-utilization` is deliberately
exposed because the usable value here is model-dependent and tight.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import random
import statistics
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))


@dataclasses.dataclass
class Result:
    backend: str
    model: str
    requests: int
    prompt_tokens: int
    output_tokens: int
    wall_s: float
    tpot_ms_p50: float
    tpot_ms_p90: float
    tpot_ms_p99: float
    ttft_ms_p50: float
    output_tok_per_s: float
    total_tok_per_s: float
    slo_attainment: dict[str, float]
    tpot_source: str
    max_model_len: int
    gpu_memory_utilization: float
    note: str = ""


def build_trace(n: int, prompt_lo: int, prompt_hi: int, seed: int) -> list[str]:
    """Synthetic prompts of controlled token-ish length.

    Word-count is a proxy for token count, which is fine here: the point is a
    spread of prompt lengths so the batch is ragged, not an exact token budget.
    """
    rng = random.Random(seed)
    vocab = ("the quick brown fox jumps over lazy dogs while carefully "
             "measuring latency throughput and memory bandwidth on a small gpu").split()
    out = []
    for _ in range(n):
        length = rng.randint(prompt_lo, prompt_hi)
        out.append(" ".join(rng.choice(vocab) for _ in range(length)))
    return out


def run(args) -> Result:
    from vllm import LLM, SamplingParams

    stats = None
    if args.backend == "ours":
        # Patch the platform hook *before* the engine is constructed. The V1
        # engine normally runs in a child process, so the caller must also set
        # VLLM_ENABLE_V1_MULTIPROCESSING=0 for this to reach the worker; main()
        # does that.
        import integration.vllm_backend as vb
        qualname = vb.install()
        stats = vb.STATS
        print(f"  [backend] patched attention backend -> {qualname}")

    llm_kwargs = dict(
        model=args.model,
        dtype="float16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
        swap_space=0,
        disable_log_stats=True,
    )
    llm = LLM(**llm_kwargs)

    prompts = build_trace(args.requests, args.prompt_lo, args.prompt_hi, args.seed)
    # Fixed output length: TPOT is only comparable across backends when every
    # request decodes the same number of steps.
    params = SamplingParams(temperature=0.0, max_tokens=args.output_tokens,
                            ignore_eos=True)

    llm.generate(prompts[: min(4, len(prompts))], params)  # warm up / compile

    if stats is not None:
        stats.update(decode_calls=0, decode_tokens=0, delegated_calls=0)

    t0 = time.perf_counter()
    outs = llm.generate(prompts, params)
    wall = time.perf_counter() - t0

    if stats is not None:
        print(f"  [backend] our decode kernel ran {stats['decode_calls']:,} times "
              f"over {stats['decode_tokens']:,} tokens; "
              f"{stats['delegated_calls']:,} calls delegated to vLLM "
              f"(prefill and non-decode batches)")
        if stats["decode_calls"] == 0:
            raise RuntimeError(
                "our backend was installed but never executed a decode -- "
                "refusing to report numbers that did not come from it")

    tpots_ms: list[float] = []
    ttfts_ms: list[float] = []
    n_out = 0
    n_prompt = 0
    for o in outs:
        n_prompt += len(o.prompt_token_ids or [])
        gen = o.outputs[0]
        n_out += len(gen.token_ids)
        m = getattr(o, "metrics", None)
        if m is None or m.first_token_time is None or m.last_token_time is None:
            continue
        if len(gen.token_ids) > 1:
            tpots_ms.append(
                (m.last_token_time - m.first_token_time) / (len(gen.token_ids) - 1) * 1e3
            )
        if m.arrival_time is not None:
            ttfts_ms.append((m.first_token_time - m.arrival_time) * 1e3)

    tpot_source = "per-request metrics"
    if not tpots_ms:
        # vLLM's V1 engine does not populate RequestOutput.metrics, so isolate
        # decode from prefill with a two-point measurement instead: run the same
        # trace generating 1 token, then N, and difference them.
        #
        #     TPOT = (wall(N) - wall(1)) / (N - 1)
        #
        # Everything that is not per-decode-step -- prefill, scheduling,
        # tokenization, engine startup per call -- appears in both terms and
        # cancels. It yields one aggregate TPOT rather than a distribution, so
        # the percentiles below collapse to that single value and are reported
        # as such rather than invented.
        tpot_source = "two-point (wall(N) - wall(1)) / (N - 1)"
        params1 = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)
        t1 = time.perf_counter()
        llm.generate(prompts, params1)
        wall1 = time.perf_counter() - t1
        steps = args.output_tokens - 1
        if steps < 1:
            raise RuntimeError("--output-tokens must be >= 2 for the two-point method")
        per_step_s = (wall - wall1) / steps
        # Per *step*, the batch decodes many sequences at once; TPOT is the
        # per-token latency a single request sees, which is the step time.
        tpots_ms = [per_step_s * 1e3]
        print(f"  [two-point] wall(1 tok)={wall1:.2f}s  wall({args.output_tokens} tok)="
              f"{wall:.2f}s  -> {per_step_s*1e3:.2f} ms/step")

    s = sorted(tpots_ms)

    def pct(p: float) -> float:
        return s[min(len(s) - 1, int(p * (len(s) - 1)))]

    return Result(
        backend=args.backend or "vllm-default",
        model=args.model,
        requests=len(prompts),
        prompt_tokens=n_prompt,
        output_tokens=n_out,
        wall_s=wall,
        tpot_ms_p50=statistics.median(s),
        tpot_ms_p90=pct(0.90),
        tpot_ms_p99=pct(0.99),
        ttft_ms_p50=statistics.median(sorted(ttfts_ms)) if ttfts_ms else float("nan"),
        output_tok_per_s=n_out / wall,
        total_tok_per_s=(n_out + n_prompt) / wall,
        slo_attainment={
            f"{x:g}ms": sum(1 for t in s if t <= x) / len(s)
            for x in [float(v) for v in args.slo.split(",") if v]
        },
        tpot_source=tpot_source,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct",
                    help="needs GQA group 4 to stay comparable to the kernel sweep")
    ap.add_argument("--backend", default=None, choices=[None, "ours"],
                    help="'ours' installs the PagedAttn Triton decode backend; "
                         "omit for vLLM's own")
    ap.add_argument("--requests", type=int, default=32)
    ap.add_argument("--prompt-lo", type=int, default=64)
    ap.add_argument("--prompt-hi", type=int, default=512)
    ap.add_argument("--output-tokens", type=int, default=128)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--enforce-eager", action="store_true",
                    help="disable CUDA graphs; useful to isolate launch overhead")
    ap.add_argument("--slo", default="10,20,50,100")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/vllm_e2e.json")
    args = ap.parse_args()

    import os
    # Always in-process, for both backends. Our monkeypatch only reaches the
    # engine when it runs here rather than in a spawned worker -- and running
    # the baseline the same way is the difference between a comparison and a
    # confound.
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    import torch
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable -- see integration/README.md 'driver trap'")
    torch.randn(8, device="cuda").sum().item()  # a real launch, not just an import

    r = run(args)
    print()
    print(f"backend            {r.backend}")
    print(f"attention          {'PagedAttn Triton (ours)' if r.backend == 'ours' else 'vLLM default'}")
    print(f"model              {r.model}")
    print(f"requests           {r.requests}  ({r.prompt_tokens} prompt + "
          f"{r.output_tokens} generated tokens)")
    print(f"wall               {r.wall_s:.2f} s")
    if r.tpot_source.startswith("two-point"):
        print(f"TPOT               {r.tpot_ms_p50:.2f} ms  (aggregate; no "
              "per-request distribution, so p90/p99 are not reported)")
    else:
        print(f"TPOT               p50 {r.tpot_ms_p50:.2f}  p90 {r.tpot_ms_p90:.2f}  "
              f"p99 {r.tpot_ms_p99:.2f} ms")
    print(f"TPOT source        {r.tpot_source}")
    print(f"TTFT p50           {r.ttft_ms_p50:.1f} ms")
    print(f"output throughput  {r.output_tok_per_s:,.0f} tok/s")
    print(f"total throughput   {r.total_tok_per_s:,.0f} tok/s")
    print("SLO attainment     " + "  ".join(f"{k}:{v*100:.0f}%"
                                            for k, v in r.slo_attainment.items()))

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if out.exists():
        try:
            existing = json.loads(out.read_text(encoding="utf-8")).get("runs", [])
        except Exception:
            existing = []
    existing.append(dataclasses.asdict(r))
    out.write_text(json.dumps({"runs": existing}, indent=2), encoding="utf-8")
    print(f"\nwrote {out}  ({len(existing)} run(s) recorded)")


if __name__ == "__main__":
    main()
