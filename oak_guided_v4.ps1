# TORSO YAW V4 -- guided LIVE OAK-D capture, SPOKEN.
#
# The subject stands ~2 m from the camera and cannot read a console, so every instruction is spoken
# aloud through the Windows speech synthesiser and each block boundary is marked with a beep.
# Writes wall-clock (UTC epoch seconds) block boundaries to oak_v4_evidence/guided_marks.json so the
# Unity trace and the sidecar log can both be cut per motion.
#
#   powershell -ExecutionPolicy Bypass -File oak_guided_v4.ps1

param([string]$Out = "oak_v4_evidence\guided_marks.json")

Add-Type -AssemblyName System.Speech
$voice = New-Object System.Speech.Synthesis.SpeechSynthesizer
$voice.Rate = 0
$voice.Volume = 100

# name, seconds, spoken instruction
$BLOCKS = @(
  @{n="rest";        s=8;  say="Stand still. Face the camera. Arms relaxed."},
  @{n="slow_yaw";    s=14; say="Turn your torso slowly left. Then slowly back to centre. Then slowly right."},
  @{n="fast_yaw";    s=12; say="Now turn fast. Left, right, left, right. Keep your feet still."},
  @{n="turn_lr";     s=14; say="Face left. Hold. Back to centre. Face right. Hold. Back to centre."},
  @{n="repeat_lr";   s=14; say="Turn left and right repeatedly, about one full cycle every second."},
  @{n="yaw_90";      s=12; say="Turn ninety degrees to your left. Hold it. Then return to centre."},
  @{n="yaw_180";     s=12; say="Turn all the way around. Show your back to the camera. Hold. Then return."},
  @{n="arm_only";    s=12; say="Keep your torso still. Move both arms only. Raise them, lower them, wave."},
  @{n="torso_only";  s=12; say="Keep your arms still at your sides. Rotate your torso left and right."},
  @{n="torso_arms";  s=12; say="Now both together. Rotate your torso and move your arms."},
  @{n="fast_both";   s=12; say="Finally, fast. Rotate your torso quickly while waving both arms quickly."}
)

function Beep-Start { [console]::beep(880,150); [console]::beep(1320,220) }
function Beep-End   { [console]::beep(660,180) }
function Now-Epoch  { [double]([DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()) / 1000.0 }

$total = ($BLOCKS | Measure-Object -Property s -Sum).Sum
Write-Host ("=" * 78)
Write-Host " GUIDED OAK-D TORSO YAW CAPTURE -- $($BLOCKS.Count) blocks, ~$total s of motion"
Write-Host ("=" * 78)

$voice.Speak("Torso tracking capture. Stand about two metres back, full body in frame. Starting in ten seconds.")
for ($i = 10; $i -gt 0; $i--) {
    Write-Host -NoNewline "`r  starting in $i s ... "
    Start-Sleep -Seconds 1
}
Write-Host "`r  GO                            "

$marks = @()
$tStart = Now-Epoch
foreach ($b in $BLOCKS) {
    $voice.Speak("Next. " + $b.say)
    Beep-Start
    $t0 = Now-Epoch
    for ($r = $b.s; $r -gt 0; $r--) {
        Write-Host -NoNewline ("`r  {0,-12} {1,2}s remaining " -f $b.n, $r)
        Start-Sleep -Seconds 1
    }
    $t1 = Now-Epoch
    Beep-End
    $marks += [pscustomobject]@{ name = $b.n; t0 = [math]::Round($t0,3); t1 = [math]::Round($t1,3); instr = $b.say }
    Write-Host ("`r  {0,-12} done ({1:N1}s)              " -f $b.n, ($t1 - $t0))
}

$voice.Speak("Capture complete. Thank you.")
$dir = Split-Path -Parent $Out
if ($dir -and -not (Test-Path $dir)) { New-Item -ItemType Directory -Force $dir | Out-Null }
[pscustomobject]@{ start = [math]::Round($tStart,3); end = [math]::Round((Now-Epoch),3); blocks = $marks } |
    ConvertTo-Json -Depth 5 | Out-File -Encoding utf8 $Out
Write-Host ("=" * 78)
Write-Host " DONE -> $Out"
Write-Host ("=" * 78)
