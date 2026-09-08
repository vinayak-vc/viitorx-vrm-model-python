@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

REM ============================================================================
REM  P0 ACCEPTANCE CAPTURE  (OAK-D -> Python -> UDP -> Unity -> VRM avatar)
REM
REM  Runs the REAL OAK-D pipeline with the P0 settings + full diagnostic logging,
REM  then prints the acceptance evidence tables (analyze_capture.py).
REM
REM  REQUIRES A HUMAN SUBJECT IN FRAME - the occlusion/reacquire/spike tests
REM  cannot be produced without someone performing the motions.
REM
REM  BEFORE RUNNING:
REM    1) Unity: open Scenes/Bootstrap.unity, confirm AppBootstrap.pipelineLogging = ON,
REM       press PLAY, and make sure the VRM avatar is visible.
REM    2) Stand ~2.0-2.5 m from the OAK-D, full body in frame (check the preview).
REM
REM  THE SCRIPT (run each block, ~15 s each, keep moving through them in order):
REM     A  stand still ...................... 15 s
REM     B  walk in place, normal speed ...... 15 s
REM     C  fast arm movement / swings ....... 15 s
REM     D  fast leg movement / kicks ........ 15 s
REM     E  LEFT HAND behind torso ........... hold 5 s, release, repeat x3
REM     F  arms crossed (hides both elbows) . hold 5 s, release, repeat x3
REM     G  turn side-on (hides left limbs) .. hold 5 s, release, repeat x3
REM     H  hide LEFT LEG behind right ....... hold 5 s, release, repeat x3
REM     I  LONG occlusion: left hand behind back for 10 s, then release
REM     J  turning 360 slowly ............... 15 s
REM
REM  Press q or ESC in the preview window to STOP early.
REM ============================================================================

set "MODEL=%~1"
if "%MODEL%"=="" set "MODEL=..\..\..\SentisModel\rtmw3d-x.onnx"
set "LOGDIR=pipeline_logs"
REM P0 shipping settings. --arm-max-jump / --leg-max-jump default to 0.35 (P0-2 cap).
set "SIDECAR_ARGS=--min-cutoff 0.5 --beta 0.4 --depth-min-cutoff 0.3 --depth-beta 0.1 --max-hold-frames 8"
REM Auto-stop seconds (0 = manual q/ESC). 180 s covers blocks A-J above.
set "SECONDS=180"

set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

if not exist "%MODEL%" (
  echo [ERROR] Model not found: "%MODEL%"
  echo   Pass the rtmw3d-x.onnx path as the first argument.
  pause & exit /b 1
)

set "SECS_ARG="
if not "%SECONDS%"=="0" set "SECS_ARG=--seconds %SECONDS%"

echo ==================================================================
echo  P0 ACCEPTANCE CAPTURE  (%SECONDS% s)
echo   Unity must be in PLAY with pipelineLogging = ON.
echo   Work through blocks A-J printed above / in this file.
echo  Model: %MODEL%
echo  Logs : %LOGDIR%\  (sender, holds, recv, model)
echo ==================================================================
echo.

"%PY%" wholebody_udp_sender.py --model "%MODEL%" --log-dir "%LOGDIR%" --show %SIDECAR_ARGS% %SECS_ARG%

echo.
echo [capture done] STOP Unity play mode now, then the analysis runs.
echo ==================================================================
"%PY%" analyze_capture.py --dir "%LOGDIR%" --label "P0 ACCEPTANCE (real OAK-D + Unity + VRM)"
echo ==================================================================
echo.
echo Raw logs: %LOGDIR%\sender_log.jsonl  holds_log.jsonl  recv_log.jsonl  model_log.jsonl
echo Baseline: pipeline_logs_baseline_audit\   (2026-08-11 pre-P0, for the before/after table)
pause
endlocal
