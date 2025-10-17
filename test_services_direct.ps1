# PowerShell script to test services directly through kubectl exec
# This avoids port forwarding issues

Write-Host "Testing crypto trading services directly through kubectl..."

# Function to test service health
function Test-ServiceHealth {
    param(
        [string]$ServiceName,
        [string]$PodName,
        [string]$HealthPath = "/health"
    )
    
    Write-Host "Testing $ServiceName ($PodName)..."
    
    try {
        # Try to exec into the pod and test the health endpoint
        $result = kubectl exec $PodName -n crypto-trading -- curl -s http://localhost:8000$HealthPath 2>$null
        
        if ($result -and $result -notlike "*error*" -and $result -notlike "*not found*") {
            Write-Host "HEALTHY: $ServiceName"
            return $true
        } else {
            Write-Host "UNHEALTHY: $ServiceName - curl not available or error"
            return $false
        }
    }
    catch {
        Write-Host "ERROR: $ServiceName - $($_.Exception.Message)"
        return $false
    }
}

# Get current pod names
$tradeExecutorPod = (kubectl get pods -n crypto-trading | Select-String "trade-executor-real" | ForEach-Object { ($_ -split '\s+')[0] })
$tradeOrchestratorPod = (kubectl get pods -n crypto-trading | Select-String "trade-orchestrator-llm" | ForEach-Object { ($_ -split '\s+')[0] })
$llmValidationPod = (kubectl get pods -n crypto-trading | Select-String "ollama-llm-validation" | ForEach-Object { ($_ -split '\s+')[0] })
$signalGeneratorPod = (kubectl get pods -n crypto-trading | Select-String "signal-generator-real" | ForEach-Object { ($_ -split '\s+')[0] })

Write-Host "Found pods:"
Write-Host "  Trade Executor: $tradeExecutorPod"
Write-Host "  Trade Orchestrator: $tradeOrchestratorPod"
Write-Host "  LLM Validation: $llmValidationPod"
Write-Host "  Signal Generator: $signalGeneratorPod"
Write-Host ""

# Test each service
$results = @()

if ($tradeExecutorPod) {
    $results += @{Service="Trade Executor"; Healthy=(Test-ServiceHealth -ServiceName "Trade Executor" -PodName $tradeExecutorPod -HealthPath "/health")}
}

if ($tradeOrchestratorPod) {
    $results += @{Service="Trade Orchestrator"; Healthy=(Test-ServiceHealth -ServiceName "Trade Orchestrator" -PodName $tradeOrchestratorPod -HealthPath "/health")}
}

if ($llmValidationPod) {
    $results += @{Service="LLM Validation"; Healthy=(Test-ServiceHealth -ServiceName "LLM Validation" -PodName $llmValidationPod -HealthPath "/health")}
}

if ($signalGeneratorPod) {
    $results += @{Service="Signal Generator"; Healthy=(Test-ServiceHealth -ServiceName "Signal Generator" -PodName $signalGeneratorPod -HealthPath "/health")}
}

Write-Host ""
Write-Host "=== SERVICE HEALTH SUMMARY ==="
foreach ($result in $results) {
    $status = if ($result.Healthy) { "HEALTHY" } else { "UNHEALTHY" }
    Write-Host "$($result.Service): $status"
}

$healthyCount = ($results | Where-Object { $_.Healthy }).Count
$totalCount = $results.Count

Write-Host ""
Write-Host "Overall Status: $healthyCount/$totalCount services healthy"

if ($healthyCount -eq $totalCount) {
    Write-Host "SUCCESS: All services are healthy!"
} else {
    Write-Host "WARNING: Some services need attention"
}

Write-Host ""
Write-Host "Alternative: Check service logs with:"
Write-Host "kubectl logs <pod-name> -n crypto-trading --tail=10"
