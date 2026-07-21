@echo off
REM Double-click launcher for the E1 harness GUI (Windows).
cd /d "%~dp0"
echo Starting E1 harness GUI...  (close this window to stop)
python gui\server.py %*
pause
