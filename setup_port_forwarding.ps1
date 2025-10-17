# PowerShell script to set up port forwarding for crypto trading services
# This script properly handles background processes in PowerShell

Write-Host "Setting up port forwarding for crypto trading services..."

# Function to start port forwarding in background
function Start-PortForward {
    param(
        [string]$ServiceName,
        [string]$PodName,
        [int]$LocalPort,
        [int]$RemotePort,
        [string]$Namespace = "crypto-trading"
    )
    
    Write-Host "Starting port forward for $ServiceName on port $LocalPort..."
    
    # Start the port forward as a background job
    $job = Start-Job -ScriptBlock {
        param($pod, $local, $remote, $ns)
        kubectl port-forward $pod $local`:$remote -n $ns
    } -ArgumentList $PodName, $LocalPort, $RemotePort, $Namespace
    
    Write-Host "Port forward job started for $ServiceName (Job ID: $($job.Id))"
    return $job
}

# Get current pod names
Write-Host "Getting current pod names..."

$tradeExecutorPod = (kubectl get pods -n crypto-trading | Select-String "trade-executor-real" | ForEach-Object { ($_ -split '\s+')[0] })
$tradeOrchestratorPod = (kubectl get pods -n crypto-trading | Select-String "trade-orchestrator-llm" | ForEach-Object { ($_ -split '\s+')[0] })
$llmValidationPod = (kubectl get pods -n crypto-trading | Select-String "ollama-llm-validation" | ForEach-Object { ($_ -split '\s+')[0] })
$signalGeneratorPod = (kubectl get pods -n crypto-trading | Select-String "signal-generator-real" | ForEach-Object { ($_ -split '\s+')[0] })

Write-Host "Found pods:"
Write-Host "  Trade Executor: $tradeExecutorPod"
Write-Host "  Trade Orchestrator: $tradeOrchestratorPod"
Write-Host "  LLM Validation: $llmValidationPod"
Write-Host "  Signal Generator: $signalGeneratorPod"

# Start port forwarding jobs
$jobs = @()

if ($tradeExecutorPod) {
    $jobs += Start-PortForward -ServiceName "Trade Executor" -PodName $tradeExecutorPod -LocalPort 8024 -RemotePort 8024
}

if ($tradeOrchestratorPod) {
    $jobs += Start-PortForward -ServiceName "Trade Orchestrator" -PodName $tradeOrchestratorPod -LocalPort 8023 -RemotePort 8023
}

if ($llmValidationPod) {
    $jobs += Start-PortForward -ServiceName "LLM Validation" -PodName $llmValidationPod -LocalPort 8050 -RemotePort 8050
}

if ($signalGeneratorPod) {
    $jobs += Start-PortForward -ServiceName "Signal Generator" -PodName $signalGeneratorPod -LocalPort 8000 -RemotePort 8000
}

Write-Host ""
Write-Host "Port forwarding setup complete!"
Write-Host "Active jobs:"
Get-Job

Write-Host ""
Write-Host "To check job status: Get-Job"
Write-Host "To stop all jobs: Get-Job | Stop-Job"
Write-Host "To see job output: Receive-Job -Id <JobId>"

# Wait a moment for services to be ready
Write-Host ""
Write-Host "Waiting for services to be ready..."
Start-Sleep -Seconds 3

# Test connectivity
Write-Host "Testing service connectivity..."

$services = @(
    @{Name="Trade Executor"; Port=8024; Path="/health"},
    @{Name="Trade Orchestrator"; Port=8023; Path="/health"},
    @{Name="LLM Validation"; Port=8050; Path="/health"},
    @{Name="Signal Generator"; Port=8000; Path="/health"}
)

foreach ($service in $services) {
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:$($service.Port)$($service.Path)" -UseBasicParsing -TimeoutSec 5
        Write-Host "✅ $($service.Name): Connected (Status: $($response.StatusCode))"
    }
    catch {
        Write-Host "❌ $($service.Name): Not accessible - $($_.Exception.Message)"
    }
}

Write-Host ""
Write-Host "Port forwarding setup complete!"
