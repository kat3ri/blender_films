# Installs Blender 4.5 LTS and verifies it runs headless.
# Usage:  powershell -ExecutionPolicy Bypass -File scripts\setup_blender.ps1

$ErrorActionPreference = "Stop"
$packageId = "BlenderFoundation.Blender.LTS.4.5"

Write-Output "Installing $packageId ..."
# --source winget is required: the msstore source fails certificate validation
# here and winget then refuses to disambiguate.
winget install --id $packageId --source winget --silent `
    --accept-package-agreements --accept-source-agreements

$candidates = @(
    "C:\Program Files\Blender Foundation\Blender 4.5\blender.exe",
    "C:\Program Files\Blender Foundation\Blender\blender.exe"
)
$blender = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $blender) {
    $blender = (Get-Command blender -ErrorAction SilentlyContinue).Source
}
if (-not $blender) {
    throw "Blender installed but could not be located. Set PREVIS_BLENDER to its path."
}

Write-Output "Found: $blender"
Write-Output "Verifying headless operation ..."
& $blender --background --factory-startup --python-expr `
    "import bpy, sys; print('blender ' + bpy.app.version_string); print('python ' + sys.version.split()[0])"

Write-Output ""
Write-Output "Ready. Try:  python -m previs.cli render shots\examples\synth_test_shot.json"
