# Deploy the API + dashboard to Azure Container Apps (works with Azure for Students credit).
#
# One-time setup (see README): Docker Desktop running, Docker Hub account, Azure CLI installed, `az login`.
# Usage, from the repo root:
#   powershell -ExecutionPolicy Bypass -File deploy\azure_deploy.ps1 -DockerUser <your-docker-hub-username>
# Re-run it after code changes with a new -Tag (v2, v3, ...) to update the live app.

param(
    [Parameter(Mandatory = $true)][string]$DockerUser,
    [string]$Tag = "v1",
    [string]$App = "rwanda-weather",
    [string]$Group = "rwanda-weather-rg",
    [string]$Location = "southafricanorth",
    [string]$Cpu = "2.0",
    [string]$Memory = "4.0Gi"
)

$ErrorActionPreference = "Stop"
$image = "docker.io/$DockerUser/rwanda-weather-api:$Tag"

Write-Host "1/4 Building image $image" -ForegroundColor Cyan
docker build -t $image .
Write-Host "2/4 Pushing image to Docker Hub" -ForegroundColor Cyan
docker push $image

Write-Host "3/4 Preparing Azure (first run takes a few minutes)" -ForegroundColor Cyan
az extension add --name containerapp --upgrade --only-show-errors
az provider register --namespace Microsoft.App --wait
az provider register --namespace Microsoft.OperationalInsights --wait
az group create --name $Group --location $Location --only-show-errors | Out-Null

Write-Host "4/4 Creating or updating the container app" -ForegroundColor Cyan
az containerapp up --name $App --resource-group $Group --location $Location `
    --image $image --ingress external --target-port 8000 --only-show-errors
az containerapp update --name $App --resource-group $Group `
    --cpu $Cpu --memory $Memory --min-replicas 1 --max-replicas 1 --only-show-errors | Out-Null

$fqdn = az containerapp show --name $App --resource-group $Group --query properties.configuration.ingress.fqdn -o tsv
Write-Host "`nLive app: https://$fqdn" -ForegroundColor Green
Write-Host "API docs: https://$fqdn/docs" -ForegroundColor Green
Write-Host "The model takes about a minute to load after each deploy."
