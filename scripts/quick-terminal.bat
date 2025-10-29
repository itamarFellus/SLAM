@echo off
REM Quick Terminal Setup for Cursor AI
REM This batch file opens an optimized PowerShell session

echo Starting optimized terminal for Cursor AI...
echo.

REM Set environment variables for better performance
set PYTHONUNBUFFERED=1
set PYTHONDONTWRITEBYTECODE=1

REM Start PowerShell with optimized settings
powershell.exe -NoExit -ExecutionPolicy Bypass -Command "& { $Host.UI.RawUI.WindowTitle = 'Cursor Terminal - Quick Setup'; $ConfirmPreference = 'None'; Write-Host 'Terminal ready for fast workflow!' -ForegroundColor Green; }"

pause
