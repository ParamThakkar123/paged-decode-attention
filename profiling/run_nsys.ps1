# Nsight Systems: timeline view -- launch gaps, kernel/CPU overlap, and how much
# of a decode step is actually spent inside the attention kernel.
#
#   powershell -ExecutionPolicy Bypass -File profiling/run_nsys.ps1
#
# Nsight Systems needs no special counter permissions, so this works where ncu
# may not.

param(
    [string]$OutDir = "results/nsys",
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$nsys = "nsys"
if (-not (Get-Command $nsys -ErrorAction SilentlyContinue)) {
    $cand = Get-ChildItem "C:\Program Files\NVIDIA Corporation\Nsight Systems*\target-windows-x64\nsys.exe" -ErrorAction SilentlyContinue |
            Select-Object -First 1
    if (-not $cand) { throw "nsys not found on PATH or under Program Files" }
    $nsys = $cand.FullName
}
Write-Host "using $nsys"

$configs = @(
    @{ name = "triton-b1-s16384";         impl = "triton";         batch = 1;  seq = 16384 },
    @{ name = "triton-b1-s16384-nosplit"; impl = "triton_nosplit"; batch = 1;  seq = 16384 },
    @{ name = "cuda-b1-s16384";           impl = "cuda";           batch = 1;  seq = 16384 },
    @{ name = "triton-b32-s4096";         impl = "triton";         batch = 32; seq = 4096  }
)

foreach ($c in $configs) {
    $rep = Join-Path $OutDir $c.name
    Write-Host "`n=== nsys $($c.name) ==="
    # `wddm` rather than `osrt`: osrt is not a valid trace target on Windows,
    # and the WDDM queue is exactly what we want to see here -- it is where the
    # per-launch host overhead in README section 5.1 actually lives.
    #
    # Without an elevated shell nsys prints "Wddm trace requires administrative
    # privileges, disabling" and drops that track (along with CPU sampling and
    # context switches). The CUDA and NVTX tracks still record, which is enough
    # for the launch-gap timeline; run from an admin prompt to get the WDDM queue.
    & $nsys profile `
        --trace=cuda,nvtx,wddm `
        --cuda-memory-usage=true `
        --force-overwrite=true `
        --output $rep `
        $Python profiling/profile_target.py --impl $($c.impl) --batch $($c.batch) --seqlen $($c.seq) --iters 20
    if ($LASTEXITCODE -ne 0) { Write-Warning "nsys failed for $($c.name) (exit $LASTEXITCODE)" }
}

# Text summaries so the README numbers are reproducible without the GUI.
foreach ($c in $configs) {
    $rep = Join-Path $OutDir "$($c.name).nsys-rep"
    if (Test-Path $rep) {
        & $nsys stats --report cuda_gpu_kern_sum --format csv --force-export=true `
            --output (Join-Path $OutDir $c.name) $rep 2>$null | Out-Null
    }
}

Write-Host "`nreports in $OutDir  (open the .nsys-rep files in Nsight Systems)"
