# Paged-KV GQA decode attention — a Triton kernel inside vLLM

A single-token **decode** attention kernel for a paged KV cache with
grouped-query attention, written in Triton and integrated into vLLM as a custom
V1 attention backend, where it serves every decode step of a real model.

> **Scope of this repo: real-model runs only.**
>
> Every number below comes from a real model with real weights —
> Qwen2.5-0.5B-Instruct through a real vLLM engine. Kernel microbenchmarks on a
> synthesized KV cache have been removed, along with the Nsight profiler
> captures, which profiled that same synthetic workload.
>
> **The headline is parity, not a win.** End-to-end token latency with our
> attention backend is the same as vLLM's own, within this machine's noise.
> Section 4 explains why that is the expected result and what would move it.

---

## 1. Hardware and software

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU (GA107, **sm_86**, **16 SMs**, **4.0 GiB**, 128-bit GDDR6) |
| Driver | 561.19 (CUDA 12.6) |
| Host (kernel dev) | Windows 11 Pro 22631, Python 3.13.9, PyTorch 2.12.0+cu126, triton-windows 3.8.0 |
| vLLM host | WSL2 Ubuntu 22.04, **vLLM 0.9.2**, torch 2.7.0+cu126 |
| Model | **Qwen/Qwen2.5-0.5B-Instruct** — 14 q heads / 2 KV heads, head_dim 64, GQA group 7 |

4 GiB is the constraint that shapes this project. vLLM has no Windows build, so
the engine runs under WSL2 on the same GPU (passthrough verified). Qwen2.5-0.5B
is the model used because it is ungated and small enough that a KV pool for 24
concurrent requests still fits alongside the weights at
`--gpu-memory-utilization 0.70`.

---

## 2. The kernel

`pagedattn/triton_decode.py`. One kernel with three `constexpr` switches:

  * `PER_PAGE_BT` — load the block table once per page, not once per token
  * `SPLIT_KV` — partition the KV range across CTAs (FlashDecoding), then reduce
  * `KV_DTYPE` — fp16, fp8_e5m2, or int8 with per-(token, head) scales

One program per `(batch, kv_head[, split])`. All query heads sharing a KV head
run in the same program, so each K/V byte is read from DRAM once and reused
`group` times out of registers — the reason GQA decode is worth a dedicated
kernel at all.

**Layout is vLLM-V1 / FlashInfer NHD** —
`[num_blocks, block_size, num_kv_heads, head_dim]` — chosen so no conversion is
needed anywhere in the integration. `head_dim` varies fastest, so a
`(block, token, head)` row is contiguous and the loads coalesce.

Group and `head_dim` are `constexpr`, so the kernel specializes on demand and
serves any GQA shape without an ahead-of-time instantiation list.

---

## 3. The vLLM backend

`integration/vllm_backend.py` subclasses vLLM's `TritonAttentionBackend` and
overrides **only** `forward`:

```
pure decode  (max_query_len == 1)  ->  our kernel
anything else                      ->  super().forward()
```

Building V1 attention metadata correctly — `query_start_loc`, `slot_mapping`,
block tables, cascade and local-attention variants, CUDA-graph capture paths —
is the version-sensitive part, and inheriting it is both less code and fewer
ways to be wrong.

Because the cache layout already matches, `kv_cache.unbind(0)` yields exactly
the tensors the kernel takes. No conversion, no copy.

`VLLM_ATTENTION_BACKEND` can only name vLLM's built-in `_Backend` enum members,
so an out-of-tree class cannot be selected through it. The real extension point
is one level down: vLLM asks `current_platform.get_attn_backend_cls(...)` for a
qualified name string and imports it with `resolve_obj_by_qualname`. `install()`
patches that one classmethod.

**It verifiably runs** — from [`results/vllm_ab_run.log`](results/vllm_ab_run.log):

```
=== run 1 ours ===
  [backend] our decode kernel ran 2,280 times over 54,720 tokens; 24 calls delegated to vLLM (prefill and non-decode batches)
  [two-point] wall(1 tok)=0.05s  wall(96 tok)=1.88s  -> 19.31 ms/step
```

2,280 = 24 layers × 95 decode steps. The 24 delegated calls are the single
prefill pass, one per layer. `bench_vllm.py` **raises if `decode_calls == 0`** —
a backend that silently falls back would otherwise produce a perfectly plausible
table measuring vLLM's own kernel.

**Deliberate scope limits.** Prefill, mixed prefill+decode batches, fp8 KV
caches, sliding-window and ALiBi layers and soft-capped logits are all delegated
back to vLLM. This is a single-query decode kernel with no causal mask across a
query tile and no positional-bias support; pretending otherwise would produce
wrong numbers rather than an error.

---

## 4. Result: end-to-end TPOT

Qwen2.5-0.5B-Instruct, 24 requests × 96 output tokens, **12 paired runs** — 7
with vLLM's backend first, 5 with ours first, so run-order bias cancels rather
than accumulating.

| | median TPOT | range |
|---|---:|---:|
| vLLM's own attention | 15.93 ms | 14.4 – 21.2 ms |
| ours | 15.70 ms | 13.9 – 25.9 ms |
| **median paired difference** | **−1.8 %** | −33 % to +80 % |

Read the range column before the median one. Two *vLLM-default* runs came in at
**96.6 ms and 91.1 ms** against a 15 ms median — 6× outliers that are the
machine, not the backend (this laptop has sporadic multi-second stalls; the same
effect produced the +80 % pair on our side). Those two are excluded above; with
them the median difference is −2.6 %, the same answer. Either way the spread
swamps the signal: anything under roughly ±10 % is unresolvable here, so the
honest claim is **parity**.

Raw data: [`results/vllm_e2e.json`](results/vllm_e2e.json) (forward order),
[`results/vllm_e2e_rev.json`](results/vllm_e2e_rev.json) (reversed), and the
console log [`results/vllm_ab_run.log`](results/vllm_ab_run.log).
`python bench/analyze_ab.py` reproduces the table from the JSON.

### Why parity is the expected result

Measured at this model's actual decode shape — 14/2/64, batch 24, ~300–500
tokens of context — the kernel takes **29–39 µs per call**, so all 24 layers of
attention come to **0.7–0.9 ms of a ~15.7 ms decode step: about 5 %.** Even
making attention *free* would move TPOT by 5 %, at the edge of this machine's
noise floor. A 20 % kernel win moves it by 1 %.

Attention only dominates a decode step when the KV cache is large relative to
the weights: more KV heads, longer contexts, bigger batches. Qwen2.5-0.5B with
2 KV heads and ~300-token prompts is the far corner from that — it is the model
that *fits in 4 GiB*, not the model where this kernel would matter.

That is the useful finding, and it is not one a microbenchmark can produce:
**choosing where a kernel matters is a serving decision, not a kernel one.**

### What would move it

A model with more KV heads and longer contexts. Llama-3.2-1B (32/8/64, GQA
group 4) has 4× the KV heads and would make attention a materially larger share
of the decode step; it needs a HuggingFace token, which is the only reason it is
not the model here.

---

## 5. Correctness

`python -m pytest tests -q` → **49 tests**, all passing.

The kernel is checked against a deliberately slow, obvious fp32 reference
(`pagedattn/reference.py`: explicit gather, no fusion) across sequence lengths
chosen to hit awkward cases on purpose — exactly one tile, one token past a
tile, a length that is not a multiple of the page size, an odd length with many
splits — plus ragged batches, group-1 MHA, three KV dtypes, three page sizes,
and fragmented-versus-sequential block tables.

These tests use randomly generated cache contents, because a numerical
correctness check does not depend on the values coming from a real model. They
verify the kernel is right; they make no performance claim.

---

## 6. Reproducing

```bash
pip install -r requirements.txt
python -m pytest tests -q          # 49 correctness tests
```

The end-to-end benchmark, from WSL2 — see
[`integration/README.md`](integration/README.md) for the environment, including
the CUDA-driver trap that produces a vLLM which imports cleanly and cannot run a
single kernel, and how to get Triton compiling without root:

```bash
cd /mnt/e/Projects/inference_benchmark

# 7 pairs with vLLM's backend first, 5 with ours first
bash integration/run_ab.sh 7 5

# pair them up and report the median paired difference
python bench/analyze_ab.py
```

Both orderings matter: the second backend in a pair always sees a warmer GPU, so
running only one direction bakes that bias into the answer. `analyze_ab.py`
reports the result with *and* without the runs this machine's stalls ruined, and
prints both rather than picking the flattering one. A single A-then-B comparison
on this hardware is worth nothing.

### Layout

```
pagedattn/
  config.py          shapes, KV dtypes, VRAM budget
  cache.py           paged cache, block tables, quantization
  reference.py       fp32 reference used by the tests
  triton_decode.py   the Triton kernel (split-KV, per-page BT, 3 KV dtypes)
integration/
  vllm_backend.py    vLLM V1 backend: decode -> our kernel, rest -> vLLM
  bench_vllm.py      a real model through the real engine
  run_ab.sh          the paired A/B both backends are measured with
  README.md          WSL2 environment setup and its traps
bench/
  analyze_ab.py      pairs the A/B runs, reports the paired difference
tests/               49 correctness tests vs an fp32 reference
results/             raw A/B run data and the console log
```

---

## 7. Status

**Done, measured on a real model:**

- Triton paged-KV GQA decode kernel — split-KV, fp16 / fp8-e5m2 / int8.
- A working vLLM V1 attention backend that **serves every decode step**:
  2,280 decode calls verified by counter, 24 delegated.
- 12 paired end-to-end runs in both orderings, with the outlier handling stated.
- 49 correctness tests against an fp32 reference.

**Honest limits:**

- **The result is parity, not a win** (§4), and §4 explains why that is expected
  for the only model that fits on this GPU.
- **One model only.** Qwen2.5-0.5B-Instruct, chosen because it fits and is
  ungated. It is also close to the worst case for showing a decode-attention
  win.
- **TPOT is a two-point measurement**, `(wall(N) − wall(1)) / (N − 1)`, because
  vLLM's V1 engine does not populate `RequestOutput.metrics`. That yields one
  aggregate number per run, so p90/p99 collapse to p50 and are reported as such
  rather than fabricated from a single sample.
- **This machine is noisy.** Two baseline runs landed at 91 ms and 97 ms against
  a 15 ms median. The noise floor is roughly ±10 %.
- **Decode only.** No prefill, no causal mask across a query tile, no ALiBi or
  soft-capping. Inside vLLM those cases are delegated back, by design.
- **No kernel-level baseline comparison.** Comparisons against FlashAttention-2,
  vLLM's PagedAttention and PyTorch SDPA, the bandwidth sweep, the
  KV-quantization measurements and the Nsight Compute / Systems captures were
  all measured on a synthesized KV cache, and have been removed from this repo
  under its real-models-only scope.
