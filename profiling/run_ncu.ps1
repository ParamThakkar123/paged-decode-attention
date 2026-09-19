# Nsight Compute: per-kernel counters for the configurations discussed in the README.
#
#   powershell -ExecutionPolicy Bypass -File profiling/run_ncu.ps1
#   powershell -ExecutionPolicy Bypass -File profiling/run_ncu.ps1 -Set full
#
# NOTE ON PERMISSIONS: on GeForce cards NVIDIA gates performance counters. If you
# see ERR_NVGPUCTRPERM, either run this from an elevated shell, or set
# NVIDIA Control Panel -> Desktop -> Developer Settings ->
# "Allow access to the GPU performance counters" to "All Users" and reboot.

param(
    [string]$Set = "detailed",
    [string]$OutDir = "results/ncu",
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

# Resolve ncu.exe, never ncu.bat. The .bat wrapper forwards its arguments
# through cmd.exe, which splits the kernel-name regex on its `|` and then tries
# to execute the right-hand side as a command ("'split_reduce' is not recognized
# as an internal or external command"). Calling the exe directly avoids cmd.
$ncu = $null
$cand = Get-ChildItem "C:\Program Files\NVIDIA Corporation\Nsight Compute*\target\*\ncu.exe" `
        -ErrorAction SilentlyContinue | Select-Object -First 1
if ($cand) {
    $ncu = $cand.FullName
} else {
    $onPath = Get-Command ncu -ErrorAction SilentlyContinue
    if ($onPath -and $onPath.Source -like "*.exe") { $ncu = $onPath.Source }
}
if (-not $ncu) { throw "ncu.exe not found; only the .bat wrapper is unusable here" }
Write-Host "using $ncu"

# The four points the README's profiling section is written against:
#   b1-s16384     the starved case that motivates split-KV
#   b1-s16384-ns  same point with splitting switched off (the "before")
#   b32-s4096     the saturated case where splitting should be a no-op
#   b1-pertoken   the per-token block-table gather (the other "before")
$configs = @(
    @{ name = "triton-b1-s16384";          impl = "triton";             batch = 1;  seq = 16384 },
    @{ name = "triton-b1-s16384-nosplit";  impl = "triton_nosplit";     batch = 1;  seq = 16384 },
    @{ name = "triton-b1-s16384-pertokbt"; impl = "triton_pertoken_bt"; batch = 1;  seq = 16384 },
    @{ name = "triton-b32-s4096";          impl = "triton";             batch = 32; seq = 4096  },
    @{ name = "cuda-b1-s16384";            impl = "cuda";               batch = 1;  seq = 16384 },
    @{ name = "cuda-b32-s4096";            impl = "cuda";               batch = 32; seq = 4096  }
)

foreach ($c in $configs) {
    $rep = Join-Path $OutDir $c.name
    Write-Host "`n=== ncu $($c.name) ==="
    # --kernel-name-base demangled + a regex keeps the replay to our kernels and
    # skips the cache-fill and RNG kernels from setup.
    # --set gives the standard sections; --metrics adds the two the README
    # quotes that `detailed` does not collect: the DRAM roofline percentage and
    # the sectors-per-request ratio that shows whether loads are coalesced.
    & $ncu --set $Set `
        --metrics gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio,sm__warps_active.avg.pct_of_peak_sustained_active `
        --kernel-name-base demangled `
        --kernel-name "regex:paged_decode|split_reduce|split_combine" `
        --launch-skip 1 --launch-count 4 `
        --export $rep --force-overwrite `
        --target-processes all `
        $Python profiling/profile_target.py --impl $($c.impl) --batch $($c.batch) --seqlen $($c.seq) --iters 3
    if ($LASTEXITCODE -ne 0) { Write-Warning "ncu failed for $($c.name) (exit $LASTEXITCODE)" }
}

# Text and CSV exports next to the .ncu-rep files. The .txt is the same content
# as the GUI's Details page, so every profiler number in the README can be
# regenerated and diffed without opening Nsight Compute.
foreach ($c in $configs) {
    $rep = Join-Path $OutDir "$($c.name).ncu-rep"
    if (Test-Path $rep) {
        & $ncu --import $rep --page details 2>$null |
            Out-File -Encoding utf8 (Join-Path $OutDir "$($c.name).txt")
        & $ncu --import $rep --page raw --csv 2>$null |
            Out-File -Encoding utf8 (Join-Path $OutDir "$($c.name).csv")
    }
}

Write-Host "`nreports in $OutDir"
Write-Host "now run:  python profiling/summarize_ncu.py $OutDir --out results/NCU.md"
