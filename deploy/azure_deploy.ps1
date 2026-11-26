# Deploy the API + dashboard to Azure Container Apps (works with Azure for Students credit).
#
# One-time setup (see README): Docker Desktop running, Docker Hub account, Azure CLI installed, `az login`.
# Usage, from the repo root:
#   powershell -ExecutionPolicy Bypass -File deploy\azure_deploy.ps1 -DockerUser <your-docker-hub-username>
# If the image is already on Docker Hub, add -SkipBuild to go straight to the Azure steps.
# Re-run it after code changes with a new -Tag (v2, v3, ...) to update the live app.

param(
    [Parameter(Mandatory = $true)][string]$DockerUser,
    [string]$Tag = "v1",
    [switch]$SkipBuild,
    [string]$App = "rwanda-weather",
    [string]$Group = "rwanda-weather-rg",
    [string]$Location = "southafricanorth",
    [string]$Cpu = "2.0",
    [string]$Memory = "4.0Gi"
)

$ErrorActionPreference = "Stop"
$image = "docker.io/$DockerUser/rwanda-weather-api:$Tag"

# native commands (docker, az) don't throw in PowerShell, so stop on the first failure ourselves
function Invoke-Step([string]$What, [scriptblock]$Cmd) {
    & $Cmd
    if ($LASTEXITCODE -ne 0) {
        Write-Host "`nFAILED: $What (exit code $LASTEXITCODE). Fix the error above and run the script again." -ForegroundColor Red
        exit 1
    }
}

if ($SkipBuild) {
    Write-Host "1-2/4 Skipping build and push, using $image" -ForegroundColor Cyan
} else {
    Write-Host "1/4 Building image $image" -ForegroundColor Cyan
    Invoke-Step "docker build" { docker build -t $image . }
    Write-Host "2/4 Pushing image to Docker Hub" -ForegroundColor Cyan
    Invoke-Step "docker push" { docker push $image }
}

Write-Host "3/4 Preparing Azure (first run takes a few minutes)" -ForegroundColor Cyan
Invoke-Step "az extension add" { az extension add --name containerapp --upgrade --only-show-errors }
Invoke-Step "register Microsoft.App" { az provider register --namespace Microsoft.App --wait }
Invoke-Step "register Microsoft.OperationalInsights" { az provider register --namespace Microsoft.OperationalInsights --wait }
Invoke-Step "create resource group" { az group create --name $Group --location $Location --only-show-errors | Out-Null }

Write-Host "4/4 Creating or updating the container app" -ForegroundColor Cyan
Invoke-Step "containerapp up" {
    az containerapp up --name $App --resource-group $Group --location $Location `
        --image $image --ingress external --target-port 8000
}
Invoke-Step "set CPU, memory and replicas" {
    az containerapp update --name $App --resource-group $Group `
        --cpu $Cpu --memory $Memory --min-replicas 1 --max-replicas 1 --only-show-errors | Out-Null
}

$fqdn = az containerapp show --name $App --resource-group $Group --query properties.configuration.ingress.fqdn -o tsv
Write-Host "`nLive app: https://$fqdn" -ForegroundColor Green
Write-Host "API docs: https://$fqdn/docs" -ForegroundColor Green
Write-Host "The model takes about a minute to load after each deploy."
