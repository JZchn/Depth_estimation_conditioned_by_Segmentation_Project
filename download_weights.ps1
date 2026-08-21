# download_weights.ps1
# Windows PowerShell downloader for the two checkpoints required by the project.
# Usage:  powershell -ExecutionPolicy Bypass -File download_weights.ps1
$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force -Path checkpoints | Out-Null

$files = [ordered]@{
    "checkpoints\sam_vit_b_01ec64.pth" = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    "checkpoints\depth_anything_v2_metric_hypersim_vitb.pth" = "https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Base/resolve/main/depth_anything_v2_metric_hypersim_vitb.pth"
}

foreach ($name in $files.Keys) {
    if (Test-Path $name) {
        Write-Host "skip (already exists): $name"
        continue
    }
    Write-Host "Downloading $name ..."
    curl.exe -L -o $name $files[$name]
    if ($LASTEXITCODE -ne 0) { throw "Failed to download $name" }
}

Write-Host ""
Write-Host "All weights downloaded successfully!"
