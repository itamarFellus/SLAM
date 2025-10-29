# SLAM

A from-scratch, modular 2D SLAM framework with dynamic choice of sensors, maps, and learning models.

## Terminal Optimization for Cursor AI

If you experience terminal lingering issues in Cursor AI (where the terminal doesn't proceed automatically and requires pressing Enter), use these solutions:

### Quick Fix Options:

1. **Use VS Code Settings** (Recommended)

   - The `.vscode/settings.json` file is configured to prevent terminal lingering
   - Restart Cursor after making changes

2. **Run Terminal Setup Script**

   ```powershell
   .\scripts\setup-terminal.ps1
   ```

3. **Use Quick Terminal Batch File**
   ```cmd
   .\scripts\quick-terminal.bat
   ```

### Manual Configuration:

- Set `terminal.integrated.enablePersistentSessions` to `false`
- Disable `terminal.integrated.confirmOnExit`
- Use `-NoExit` flag with PowerShell
- Set `$ConfirmPreference = "None"` in PowerShell

These settings will significantly improve your workflow speed by preventing terminal lingering issues.
