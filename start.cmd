@echo off
REM Double-click launcher for AlphaLineage. Builds the UI, runs the app on
REM http://localhost:8000, and opens a browser tab. Quit from the in-app gear menu.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1" %*
