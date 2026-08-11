@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

REM ============================================================================
REM  One-shot pipeline capture: start the OAK whole-body sidecar WITH logging,
REM  let you record, then STOP on a keypress and print the compare report.
REM
REM  Workflow:
REM    1) In Unity: make sure AppBootstrap.pipelineLogging = ON and press PLAY.
REM    2) Run this .bat (double-click, or:  run_capture.bat  [optional model path])
REM    3) Do the motions you want to diagnose (~20-30 s). Stand ~2 m from the OAK.
REM    4) Press  q  or  ESC  in the "whole-body OAK sidecar" preview window to STOP.
REM       (the compare report then runs automatically)
REM ============================================================================

REM ---- config (edit if your paths differ) ----
set "MODEL=%~1"
if "%MODEL%"=="" set "MODEL=..\..\..\SentisModel\rtmw3d-x.onnx"
set "LOGDIR=pipeline_logs"
REM sidecar smoothing/limb-depth defaults (see ADR-020); tweak here without touching code:
set "SIDECAR_ARGS=--min-cutoff 0.7 --beta 0.4 --depth-min-cutoff 0.3 --depth-beta 0.1 --max-hold-frames 8"
REM set SECONDS to a number for auto-stop instead of pressing q (0 = manual q/ESC):
set "SECONDS=0"
REM --------------------------------------------

set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

if not exist "%MODEL%" (
  echo [ERROR] Model not found: "%MODEL%"
  echo   Pass the rtmw3d-x.onnx path as the first argument, or edit MODEL at the top.
  echo   Example:  run_capture.bat "D:\path\to\rtmw3d-x.onnx"
  pause & exit /b 1
)

set "SECS_ARG="
if not "%SECONDS%"=="0" set "SECS_ARG=--seconds %SECONDS%"

echo ==================================================================
echo  OAK whole-body sidecar + 3-stage pipeline logging
echo   - Unity must be in PLAY with pipelineLogging = ON.
echo   - Stand ~2 m from the camera; do your test motions.
echo   - Press  q  or  ESC  in the preview window to STOP (then the
echo     compare report prints automatically).
echo  Model:  %MODEL%
echo  Logs :  %LOGDIR%\
echo ==================================================================
echo.

"%PY%" wholebody_udp_sender.py --model "%MODEL%" --log-dir "%LOGDIR%" --show %SIDECAR_ARGS% %SECS_ARG%

echo.
echo [sidecar stopped] running compare_logs.py ...
echo ==================================================================
"%PY%" compare_logs.py --dir "%LOGDIR%"
echo ==================================================================
echo.
echo Raw logs: %LOGDIR%\sender_log.jsonl  recv_log.jsonl  model_log.jsonl
pause
endlocal
