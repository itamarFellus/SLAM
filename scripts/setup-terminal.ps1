# PowerShell Terminal Setup Script
# This script configures PowerShell to prevent lingering and improve workflow speed

# Set PowerShell execution policy to allow local scripts
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force

# Configure PowerShell to not wait for user input
$Host.UI.RawUI.WindowTitle = "Cursor Terminal - Optimized"

# Disable confirmation prompts
$ConfirmPreference = "None"

# Set up aliases for common commands to avoid typing delays
Set-Alias -Name ll -Value Get-ChildItem
Set-Alias -Name g -Value git
Set-Alias -Name py -Value python

# Configure PSReadLine for better performance
if (Get-Module -ListAvailable -Name PSReadLine) {
    Import-Module PSReadLine
    Set-PSReadLineOption -PredictionSource History
    Set-PSReadLineOption -PredictionViewStyle ListView
    Set-PSReadLineOption -EditMode Windows
    Set-PSReadLineOption -BellStyle None
    Set-PSReadLineOption -ShowToolTips:$false
}

# Function to clear terminal without confirmation
function Clear-Terminal {
    Clear-Host
}

# Function to exit without confirmation
function Exit-Terminal {
    exit
}

# Set up environment variables for better performance
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONDONTWRITEBYTECODE = "1"

Write-Host "Terminal optimized for Cursor AI workflow!" -ForegroundColor Green
Write-Host "Use 'Clear-Terminal' to clear screen, 'Exit-Terminal' to exit" -ForegroundColor Yellow
