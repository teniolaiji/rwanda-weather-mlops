# Runs the same Locust flood against 1, 2 and 4 API containers and saves the results.
# Usage (from the repo root, with Docker Desktop running):
#   powershell -ExecutionPolicy Bypass -File locust\run_experiments.ps1
# Optional: -Python "C:\path\to\python.exe" -Users 30 -Duration 60s

param(
    [string]$Python = "$env:USERPROFILE\anaconda3\python.exe",
    [int[]]$Containers = @(1, 2, 4),
    [int]$Users = 30,
    [int]$SpawnRate = 10,
    [string]$Duration = "60s"
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force results | Out-Null

docker compose build

foreach ($n in $Containers) {
    Write-Host "`n=== $n container(s): $Users users for $Duration ===" -ForegroundColor Cyan
    docker compose up -d --scale api=$n
    docker compose restart nginx | Out-Null          # nginx re-reads the list of containers

    # wait until every container has loaded the model
    $deadline = (Get-Date).AddMinutes(3)
    do {
        Start-Sleep -Seconds 3
        $healthy = (docker compose ps api --format "{{.Health}}" | Where-Object { $_ -eq "healthy" }).Count
        Write-Host "  healthy containers: $healthy / $n"
    } until ($healthy -ge $n -or (Get-Date) -gt $deadline)

    & $Python -m locust -f locust/locustfile.py --host http://localhost:8080 `
        --headless -u $Users -r $SpawnRate -t $Duration --csv "results/c$n" --only-summary
}

& $Python locust/summarize.py
Write-Host "`nDone. Results are in the results\ folder. Stop everything with: docker compose down" -ForegroundColor Green
