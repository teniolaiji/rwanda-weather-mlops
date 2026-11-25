# Runs the same Locust flood against the LIVE Azure app with 1, 2 and 4 container replicas.
# Usage (from the repo root, after deploy\azure_deploy.ps1):
#   powershell -ExecutionPolicy Bypass -File locust\run_cloud_experiments.ps1
# Results go to results\cloud_c1_stats.csv etc. Afterwards the app is set back to 1 replica.

param(
    [string]$Python = "$env:USERPROFILE\anaconda3\python.exe",
    [string]$App = "rwanda-weather",
    [string]$Group = "rwanda-weather-rg",
    [int[]]$Replicas = @(1, 2, 4),
    [int]$Users = 30,
    [int]$SpawnRate = 10,
    [string]$Duration = "60s"
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force results | Out-Null
$url = "https://" + (az containerapp show --name $App --resource-group $Group --query properties.configuration.ingress.fqdn -o tsv)

foreach ($n in $Replicas) {
    Write-Host "`n=== $n replica(s) on Azure: $Users users for $Duration ===" -ForegroundColor Cyan
    az containerapp update --name $App --resource-group $Group --min-replicas $n --max-replicas $n --only-show-errors | Out-Null

    # wait until Azure reports n running replicas, then give the model time to load
    $deadline = (Get-Date).AddMinutes(5)
    do {
        Start-Sleep -Seconds 10
        $running = (az containerapp replica list --name $App --resource-group $Group --query "length(@)" -o tsv)
        Write-Host "  running replicas: $running / $n"
    } until ([int]$running -ge $n -or (Get-Date) -gt $deadline)
    Start-Sleep -Seconds 60

    & $Python -m locust -f locust/locustfile.py --host $url `
        --headless -u $Users -r $SpawnRate -t $Duration --csv "results/cloud_c$n" --only-summary
}

az containerapp update --name $App --resource-group $Group --min-replicas 1 --max-replicas 1 --only-show-errors | Out-Null
& $Python locust/summarize.py --prefix cloud_c
Write-Host "`nDone. Results are in results\ (cloud_summary.md, cloud_latency_vs_containers.png)." -ForegroundColor Green
