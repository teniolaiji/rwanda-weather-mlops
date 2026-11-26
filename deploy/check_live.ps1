# Smoke test for the live Azure app: health, metrics, and one prediction per class.
# Usage, from the repo root:
#   powershell -ExecutionPolicy Bypass -File deploy\check_live.ps1
# or point it at any URL:  -Url https://something.azurecontainerapps.io

param(
    [string]$Url,
    [string]$App = "rwanda-weather",
    [string]$Group = "rwanda-weather-rg"
)

if (-not $Url) {
    $fqdn = az containerapp show --name $App --resource-group $Group --query properties.configuration.ingress.fqdn -o tsv
    if (-not $fqdn) { Write-Host "Could not find the app in Azure. Did the deploy finish?" -ForegroundColor Red; exit 1 }
    $Url = "https://$fqdn"
}
$Url = $Url.TrimEnd("/")
Write-Host "Checking $Url`n" -ForegroundColor Cyan
$ok = $true

try {
    $h = Invoke-RestMethod "$Url/health" -TimeoutSec 60
    Write-Host ("PASS  /health: {0}, model v{1}, up {2}" -f $h.status, $h.model_version, $h.uptime) -ForegroundColor Green
} catch {
    Write-Host "FAIL  /health: $($_.Exception.Message)  (the model needs about a minute to load after a deploy; wait and retry)" -ForegroundColor Red
    exit 1
}

try {
    $m = Invoke-RestMethod "$Url/metrics" -TimeoutSec 30
    Write-Host ("PASS  /metrics: accuracy {0}, macro F1 {1}" -f $m.test_metrics.accuracy, $m.test_metrics.f1_macro) -ForegroundColor Green
} catch { Write-Host "FAIL  /metrics: $($_.Exception.Message)" -ForegroundColor Red; $ok = $false }

foreach ($c in "fogsmog", "hail", "lightning", "rain", "rainbow") {
    $img = Get-ChildItem "data\retrain_demo\$c\*.jpg" | Select-Object -First 1
    $raw = curl.exe -s -m 60 -F "file=@$($img.FullName)" "$Url/predict"
    try { $p = $raw | ConvertFrom-Json } catch { $p = $null }
    if ($p -and $p.label -eq $c) {
        Write-Host ("PASS  predict {0,-9} -> {1} ({2:P1}, {3} ms)" -f $c, $p.label, $p.confidence, $p.latency_ms) -ForegroundColor Green
    } else {
        Write-Host "FAIL  predict $c -> $raw" -ForegroundColor Red; $ok = $false
    }
}

try {
    $u = Invoke-RestMethod "$Url/uploads" -TimeoutSec 30
    Write-Host ("INFO  uploads in database: {0} total, {1} waiting for retraining" -f $u.totals.total, $u.totals.pending)
} catch { Write-Host "FAIL  /uploads: $($_.Exception.Message)" -ForegroundColor Red; $ok = $false }

if ($ok) { Write-Host "`nAll checks passed. Dashboard: $Url" -ForegroundColor Green }
else { Write-Host "`nSome checks failed, see above." -ForegroundColor Red; exit 1 }
