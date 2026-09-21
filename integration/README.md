# vLLM integration

Status: **working.** Our Triton decode kernel runs inside a real vLLM engine and
serves every decode step; `bench_vllm.py` verifies this by counting calls and
**refuses to report numbers if the kernel never ran**.

    [backend] patched attention backend -> integration.vllm_backend.PagedAttnTritonBackend
    [backend] our decode kernel ran 2,280 times over 54,720 tokens;
              24 calls delegated to vLLM (prefill and non-decode batches)

2,280 = 24 layers x 95 decode steps, and the 24 delegated calls are the single
prefill pass, one per layer - exactly the split the design intends.

## Why this is not on the main platform

vLLM has no Windows build. This machine runs the kernels natively on Windows and
the vLLM work under WSL2 on the same GPU (passthrough verified: `nvidia-smi`
inside WSL sees the RTX 3050). Setup, from a WSL2 Ubuntu shell:

```bash
# one-time, needs your password:
sudo apt install -y python3.10-venv
python3 -m venv ~/vllmenv
~/vllmenv/bin/pip install vllm
```

(If you cannot use `sudo`, `python3 -m venv --without-pip ~/vllmenv` followed by
the official `get-pip.py` bootstrap works and is what this repo used - Ubuntu's
`ensurepip` is missing without the `python3.10-venv` package.)

### The driver trap - read this before installing

`pip install vllm` on this machine produces an install that imports cleanly and
cannot run a single kernel:

```
torch 2.13.0+cu130   torch.cuda.is_available() -> False
RuntimeError: The NVIDIA driver on your system is too old (found version 12060)
```

Current vLLM pins **torch 2.13+cu130**, which needs a CUDA 13 driver. This
laptop's driver is 561.19 = **CUDA 12.6**. `torch.cuda.is_available()` returning
False is easy to miss if you only check that the import succeeded - it is worth
running an actual `x @ x` on the device before trusting the environment.

Pin a vLLM release built against cu126 instead. `vllm==0.9.2` requires
`torch==2.7.0`, whose default PyPI wheel is cu126:

```bash
python3 -m venv --without-pip ~/vllm126
curl -fsSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
~/vllm126/bin/python /tmp/get-pip.py
~/vllm126/bin/pip install vllm==0.9.2

# verify for real, not just that the import worked:
~/vllm126/bin/python -c "
import torch; x = torch.randn(512, 512, device='cuda'); print('compute OK', (x@x).sum().item() == (x@x).sum().item())
import vllm; print('vllm', vllm.__version__)"
```

Two environments exist in the `Ubuntu-22.04` distro from this session:
`~/vllmenv` (vLLM 0.29.0 + FlashInfer 0.6.18, **cu130, unusable here**) and
`~/vllm126` (the cu126 rebuild - **verified working**: torch 2.7.0+cu126,
`torch.cuda.is_available()` True, a real matmul completes, vLLM 0.9.2 imports,
and both `vllm.vllm_flash_attn` and `vllm._custom_ops.paged_attention_v1` load).

A second dependency-resolution note: plain `pip install vllm==0.9.2` spent 15+
minutes backtracking through candidate versions without downloading anything.
Pinning the heavy transitive packages explicitly makes it resolve immediately:

```bash
~/vllm126/bin/pip install --only-binary :all: \
    vllm==0.9.2 torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 xformers==0.0.30
```

### Triton needs a host C compiler - and you can get one without root

This WSL image has **no `gcc`**, and Triton JITs its runtime shim through one:

```
RuntimeError: Failed to find C compiler. Please specify via CC environment variable.
```

This blocks more than our kernel: **vLLM's own engine will not start either**,
because `vllm/v1/worker/gpu_model_runner.py` imports Triton Mamba kernels at
module load. `sudo apt install build-essential python3.10-dev` fixes everything
in one line if you have the password.

Without root, two pip-installable pieces do the job, and this is what produced
the WSL2 environment these runs use:

```bash
# 1. a C compiler, from PyPI
~/vllm126/bin/pip install ziglang
mkdir -p ~/bin
cat > ~/bin/zigcc <<'EOF'
#!/bin/sh
exec "$HOME/vllm126/bin/python" -m ziglang cc "$@"
EOF
chmod +x ~/bin/zigcc

# 2. CPython headers, extracted from the Ubuntu .deb without installing it
mkdir -p ~/debs && cd ~/debs
B=http://archive.ubuntu.com/ubuntu/pool/main/p/python3.10
V=3.10.12-1%7E22.04.18      # %7E is the URL-encoded '~' in the version
curl -sSO "$B/libpython3.10-dev_${V}_amd64.deb"
curl -sSO "$B/python3.10-dev_${V}_amd64.deb"
for d in *.deb; do dpkg-deb -x "$d" ~/pyhdr; done

# 3. point Triton at both
export CC=$HOME/bin/zigcc
export CPATH=$HOME/pyhdr/usr/include/python3.10:$HOME/pyhdr/usr/include
```

`CPATH` must include the parent `usr/include` as well as the leaf: Ubuntu's
`pyconfig.h` forwards to `<x86_64-linux-gnu/python3.10/pyconfig.h>`, so the
multiarch root has to be on the search path too. zig emits one
`_POSIX_C_SOURCE macro redefined` warning; it is harmless.

Two WSL gotchas that cost time here:

- **`/tmp` is wiped when the distro restarts**, which it does whenever the last
  `wsl` session exits. Download and extract into `$HOME`, not `/tmp`.
- **A `nohup`-backgrounded process dies with the session** for the same reason.
  Run long installs in the foreground of a `wsl` invocation that stays alive.

## How it hooks in

`VLLM_ATTENTION_BACKEND` only selects among vLLM's built-in `_Backend` enum
members, so an out-of-tree class cannot be named through it. The actual
extension point is one level down: vLLM asks
`current_platform.get_attn_backend_cls(...)` for a **qualified name string** and
then imports it with `resolve_obj_by_qualname`. `install()` patches that one
classmethod to return ours.

`PagedAttnTritonBackend` **subclasses vLLM's own `TritonAttentionBackend`** and
overrides only `get_impl_cls`; `PagedAttnTritonImpl` subclasses
`TritonAttentionImpl` and overrides only `forward`. Building V1 attention
metadata correctly - `query_start_loc`, `slot_mapping`, block tables, cascade and
local-attention variants, the CUDA-graph capture paths - is the fiddly,
version-sensitive part. Inheriting it is both less code and fewer ways to be
subtly wrong.

`forward` intercepts **pure decode** (`max_query_len == 1`) and hands everything
else back to `super().forward()`: prefill, mixed prefill+decode batches, fp8 KV
caches, sliding-window and ALiBi layers, soft-capped logits. That is the design,
not a gap - our kernel is a single-query decode kernel with no causal mask across
a query tile and no positional-bias support, and pretending otherwise would
produce wrong numbers instead of an error.

No conversion is needed anywhere: vLLM allocates the V1 KV cache as
`[2, num_blocks, block_size, num_kv_heads, head_size]`, so `kv_cache.unbind(0)`
yields exactly the NHD tensors our kernel already takes. That is the payoff for
matching this layout in `pagedattn/cache.py`.

### The multiprocessing catch

The V1 engine runs in a **child process** by default, where an in-process
monkeypatch never applies. `bench_vllm.py` sets
`VLLM_ENABLE_V1_MULTIPROCESSING=0` for *both* backends - running the baseline the
same way is the difference between a comparison and a confound, and it moved the
baseline TPOT materially. For production use, register `install()`
through a `vllm.general_plugins` entry point instead; that runs in the worker.

## Two deliberate scope limits

**Decode only.** This kernel assumes exactly one query token per sequence and has
no causal masking across a query tile. Prefill is delegated. A prefill-capable
kernel is a different kernel, not a flag on this one.

**Triton only.** Group and `head_dim` are `constexpr` in the Triton kernel, so
it specializes on demand and can serve any GQA shape. That matters for a serving
backend, which cannot refuse a model shape.

## Picking a model that fits 4 GiB

The weights, the CUDA context (~300 MiB) and the KV pool all share 4 GiB.

| model | q/kv heads | head_dim | group | fp16 weights | fits? |
|---|---|---|---:|---|---|
| Qwen2.5-0.5B | 14 / 2 | 64 | 7 | ~1.0 GiB | yes - **used for section 6.1** |
| Llama-3.2-1B | 32 / 8 | 64 | 4 | ~2.5 GiB | yes, gated |
| Qwen2.5-1.5B | 12 / 2 | 128 | 6 | ~3.1 GiB | tight |
| Llama-3-8B | 32 / 8 | 128 | 4 | ~16 GiB | no |

Qwen2.5-0.5B is what the end-to-end run uses: it is ungated, and at
`gpu_memory_utilization=0.7` it leaves enough room for a KV pool big enough to
hold 24 concurrent requests. Llama-3.2-1B has the same GQA group as the benchmark
shape and would be the better comparison, but it needs a HuggingFace token.

## Running the end-to-end benchmark

```bash
cd /mnt/e/Projects/inference_benchmark

# vLLM's own attention backend -- the baseline
PYTHONPATH=$PWD ~/vllm126/bin/python integration/bench_vllm.py     --model Qwen/Qwen2.5-0.5B-Instruct --backend default     --requests 24 --output-tokens 96 --max-model-len 1536

# ours
PYTHONPATH=$PWD ~/vllm126/bin/python integration/bench_vllm.py     --model Qwen/Qwen2.5-0.5B-Instruct --backend ours     --requests 24 --output-tokens 96 --max-model-len 1536
```

Both append to `results/vllm_e2e.json`. **Run them in alternating order and pair
them up** - this machine has sporadic multi-second stalls, and a single A-then-B
comparison on it is worth nothing. The 12 pairs behind README section 6.1 were
collected 7 one way and 5 the other so run-order bias cancels; two *baseline*
runs still landed at 91 ms and 97 ms against a 15 ms median.

`--backend ours` calls `install()` before the engine is constructed, and the run
ends with the call counters:

    [backend] our decode kernel ran 2,280 times over 54,720 tokens;
              24 calls delegated to vLLM (prefill and non-decode batches)

`bench_vllm.py` **raises if `decode_calls == 0`.** A backend that silently falls
back is the single most likely failure here and it produces a completely
plausible table - of vLLM's kernel, not ours.

TPOT is computed two-point, `(wall(N) - wall(1)) / (N - 1)`, because vLLM V1 does
not populate `RequestOutput.metrics`. That yields one number per run, so p90/p99
are reported as equal to p50 rather than fabricated from a distribution that was
never measured.

`--enforce-eager` disables vLLM's CUDA graphs. Worth running once: per-launch
host overhead on WDDM is large enough that the eager-vs-graphed gap is visible at
model scale.

## What the end-to-end numbers say

TPOT is the same within noise, and that is the expected result. At this model's
decode shape (14/2/64, batch 24, ~300-500 tokens of context) our kernel takes
29-39 us per call, so 24 layers of attention is 0.7-0.9 ms of a ~15.7 ms step --
**about 5 %**. Making attention free would move TPOT by 5 %; a 20 % kernel win
moves it by 1 %. Qwen2.5-0.5B is the model that *fits in 4 GiB*, not the model
this kernel is for -- see section 4 of the main README.
