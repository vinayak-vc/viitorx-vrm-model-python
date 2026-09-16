@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0.."

REM ============================================================================
REM  F-20B - start the sidecar supervisor/watchdog with the PRODUCTION configuration.
REM
REM  This does NOT run the sidecar directly - it runs sidecar_supervisor.py, which starts
REM  wholebody_udp_sender.py itself, watches it, and restarts it with backoff if it ever dies
REM  (crash, forced kill, OAK-D unplug, DepthAI init failure). Unity does not need restarting
REM  when the sidecar restarts - see docs/F20B_SIDECAR_SUPERVISOR_WATCHDOG_2026-09-14.md.
REM
REM  Workflow:
REM    1) In Unity: Play as normal (useOakUdpTracking = true). Unity's own F-20A logic handles
REM       Live/StaleHold/StaleFailsafe/Recovering - this script only keeps the PRODUCER alive.
REM    2) Run this .bat (double-click, or from a shell: run_supervisor.bat).
REM    3) Leave it running. Close its window (or Ctrl+C) to stop the sidecar along with it.
REM  Production installation: register this .bat with Task Scheduler to run at logon - see the
REM  report's "Production Installation Procedure" for the exact command. This script does NOT
REM  register itself; that is a persistent system change and is documented, not automated.
REM ============================================================================

set "MODEL=%~1"
if "%MODEL%"=="" set "MODEL=..\..\..\SentisModel\rtmw3d-x.onnx"
set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

if not exist "%MODEL%" (
  echo [ERROR] Model not found: "%MODEL%"
  echo   Pass the rtmw3d-x.onnx path as the first argument, or edit MODEL at the top.
  pause & exit /b 1
)

echo ==================================================================
echo  F-20B sidecar supervisor - production configuration
echo   portrait ON, portrait-dir ccw, subpixel-bits 3, 127.0.0.1:8899
echo  Model:  %MODEL%
echo  Evidence: oak_v4_evidence\f20b\
echo ==================================================================
echo.

"%PY%" -u sidecar_supervisor.py --model "%MODEL%" --portrait --portrait-dir ccw --subpixel-bits 3 --host 127.0.0.1 --port 8899

echo.
echo [supervisor stopped]
pause
endlocal
