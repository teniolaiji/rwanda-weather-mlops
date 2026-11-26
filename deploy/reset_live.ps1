# Reset the live Azure app to the model in the Docker image (version 1) with an empty uploads database.
# Uploads, retrained models and the SQLite file live on the container's local disk, so restarting the
# revision gives a clean container. Useful before recording a demo.
# Usage, from the repo root:
#   powershell -ExecutionPolicy Bypass -File deploy\reset_live.ps1

param(
    [string]$App = "rwanda-weather",
    [string]$Group = "rwanda-weather-rg"
)

# Express environments don't support "revision restart", so change a harmless setting instead:
# that rolls out a brand-new container from the same image.
$stamp = Get-Date -Format "yyyyMMddHHmmss"
Write-Host "Replacing the container with a fresh one (RESET_AT=$stamp)" -ForegroundColor Cyan
az containerapp update --name $App --resource-group $Group --set-env-vars "RESET_AT=$stamp" --only-show-errors | Out-Null
if ($LASTEXITCODE -ne 0) { exit 1 }

$url = "https://" + (az containerapp show --name $App --resource-group $Group --query properties.configuration.ingress.fqdn -o tsv)
Write-Host "Waiting for the model to load..."
$deadline = (Get-Date).AddMinutes(5)
do {
    Start-Sleep -Seconds 10
    try { $h = Invoke-RestMethod "$url/health" -TimeoutSec 15 } catch { $h = $null }
} until (($h -and $h.uptime_seconds -lt 120 -and $h.model_version -eq 1) -or (Get-Date) -gt $deadline)

if ($h -and $h.model_version -eq 1) {
    $u = Invoke-RestMethod "$url/uploads"
    Write-Host ("Reset done: model v{0}, uptime {1}, uploads in database: {2}" -f $h.model_version, $h.uptime, $u.totals.total) -ForegroundColor Green
} else {
    Write-Host "The app did not come back as model v1 within 5 minutes. Check it in the browser: $url" -ForegroundColor Red
    exit 1
}
