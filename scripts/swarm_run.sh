#!/usr/bin/env bash
# OS-scheduled swarm launcher (called by Windows Task Scheduler). Runs a headless `claude -p` tick for one role.
# Roles: data-queen | model-queen | worker | worker2..N | combinator | viz | digest.
#   workerN share the worker.txt prompt but use their OWN lock -> run in PARALLEL; per-ITEM claim (in worker.txt) stops collisions.
#   EXTRA parallel roles (worker[2-9], combinator, viz) are LOAD-GUARDED: skip the tick if CPU hot / RAM low (self-throttle).
# Usage: swarm_run.sh <role>
set -uo pipefail
ROLE="${1:?role required}"
PROJ="/d/Users/ashenoy00000/.windsurf/OpenADMET-pxr-challenge"
PROMPTNAME="$ROLE"; case "$ROLE" in worker[0-9]*) PROMPTNAME="worker" ;; esac   # workerN -> worker.txt
PROMPT="$PROJ/scripts/prompts/${PROMPTNAME}.txt"
LOG="C:/pxr_work/swarm_${ROLE}.log"
LOCK="C:/pxr_work/swarm_locks/${ROLE}.lock"
cd "$PROJ" || exit 1
export PATH="/d/Users/ashenoy00000/AppData/Roaming/npm:/c/Program Files/Git/usr/bin:$PATH"
CLAUDE="$(command -v claude || echo /d/Users/ashenoy00000/AppData/Roaming/npm/claude)"
[ -f "$PROMPT" ] || { echo "[$(date '+%F %T')] no prompt $PROMPT" >>"$LOG"; exit 1; }

# atomic per-role lock: mkdir succeeds only if not already held. Stale lock (>50min) reclaimed.
if ! mkdir "$LOCK" 2>/dev/null; then
  if [ -n "$(find "$LOCK" -maxdepth 0 -mmin +50 2>/dev/null)" ]; then rm -rf "$LOCK"; mkdir "$LOCK" 2>/dev/null || exit 0; else
    echo "[$(date '+%F %T')] $ROLE tick skipped (prior tick still running)" >>"$LOG"; exit 0; fi
fi
trap 'rm -rf "$LOCK"' EXIT

# LOAD GUARD (extra parallel roles only): skip if CPU>90% or free RAM<4.5GB so we never thrash the box.
case "$ROLE" in
  worker[2-9]*|combinator|viz)
    STAT=$(powershell -NoProfile -Command "\$c=[int]((Get-Counter '\\Processor(_Total)\\% Processor Time' -EA SilentlyContinue).CounterSamples.CookedValue); \$m=[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1024); Write-Output \"\$c \$m\"" 2>/dev/null)
    CPU=${STAT%% *}; FREE=${STAT##* }
    if [ "${CPU:-0}" -gt 90 ] || [ "${FREE:-99999}" -lt 4500 ]; then
      echo "[$(date '+%F %T')] $ROLE SKIPPED (load guard: CPU ${CPU:-?}% free ${FREE:-?}MB)" >>"$LOG"; exit 0
    fi
    echo "[$(date '+%F %T')] $ROLE load ok (CPU ${CPU}% free ${FREE}MB)" >>"$LOG" ;;
esac

echo "[$(date '+%F %T')] === $ROLE tick start ===" >>"$LOG"
timeout 1500 "$CLAUDE" -p --dangerously-skip-permissions "$(cat "$PROMPT")" >>"$LOG" 2>&1
echo "[$(date '+%F %T')] === $ROLE tick end (exit $?) ===" >>"$LOG"
