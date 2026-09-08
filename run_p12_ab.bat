@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

REM ============================================================================
REM  P1-2 HUMAN A/B  —  frame freshness with a real subject (Phases 5-6)
REM
REM  Runs the SAME motion twice: once with the old FIFO queue policy, once with
REM  the P1-2 latest-frame policy, then prints the freshness + jitter comparison.
REM
REM  REQUIRES A HUMAN IN FRAME. 2.0-2.5 m, full body visible.
REM
REM  Do the SAME motion sequence in both runs, ~40 s each:
REM     0-08 s  stand still
REM     08-16 s  walk toward / away
REM     16-24 s  fast arms (waving, big swings)
REM     24-32 s  fast legs (kicks, squats)
REM     32-40 s  dancing
REM
REM  Unity does NOT need to be running for this (sidecar-side measurement).
REM ============================================================================

set "MODEL=%~1"
if "%MODEL%"=="" set "MODEL=..\..\..\SentisModel\rtmw3d-x.onnx"
set "ARGS=--min-cutoff 0.5 --beta 0.4 --depth-min-cutoff 0.3 --depth-beta 0.1 --max-hold-frames 8"
set "SECS=40"

set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"
if not exist "%MODEL%" (
  echo [ERROR] Model not found: "%MODEL%"
  pause & exit /b 1
)

echo ==================================================================
echo  P1-2 HUMAN A/B  (2 runs x %SECS% s)
echo  Do the SAME motion in both runs. Get in frame now.
echo ==================================================================
pause

echo.
echo --- RUN 1 of 2 : OLD FIFO policy (--no-latest-frame) ---
"%PY%" wholebody_udp_sender.py --model "%MODEL%" --log-dir p12h_fifo --show --seconds %SECS% %ARGS% --no-latest-frame

echo.
echo --- Get back in position for RUN 2 ---
pause
echo --- RUN 2 of 2 : P1-2 latest-frame policy ---
"%PY%" wholebody_udp_sender.py --model "%MODEL%" --log-dir p12h_latest --show --seconds %SECS% %ARGS% --latest-frame

echo.
echo ==================================================================
"%PY%" compare_p12.py
echo ==================================================================
pause
endlocal
