#!/bin/bash
# Frequent-cron monitor for the TRAIN-side ternary (coactivator) cofold on Explorer.
# Each run: pull partials -> check SLURM queue -> resubmit if died & not done -> honest-gate eval -> early-kill if redundant.
# Resumable + idempotent. Good cluster citizen: auto-cancels once the verdict is clearly in (saves GPU).
set -uo pipefail
cd /d/Users/ashenoy00000/.windsurf/OpenADMET-pxr-challenge || exit 1
HOST=explorer; RD=/scratch/shenoy.am/boltz_pxr
LOG=C:/pxr_work/coact_monitor.log; STOP=C:/pxr_work/coact_STOP
mkdir -p C:/pxr_work
ts() { date '+%Y-%m-%d %H:%M:%S'; }
say() { echo "[$(ts)] $*" | tee -a "$LOG"; }
SSH="ssh -o ConnectTimeout=25 $HOST"

say "=== monitor tick ==="

# 1. pull partials (remote glob expands remote-side)
timeout 120 scp "$HOST":"$RD"/coact_train_t'*'.npy data/processed/ 2>/dev/null && say "pulled partials" || say "scp pull skipped/failed"

# 2. local done count from merged partials
DONE=$(.venv/Scripts/python.exe -c "
import numpy as np, glob
d=np.zeros(4139,bool)
for f in glob.glob('data/processed/coact_train_t*_done.npy'):
    d |= np.load(f).astype(bool)
print(int(d.sum()))" 2>/dev/null || echo 0)
say "train ternary done: ${DONE}/4139"

# 3. queue state
INQ=$(timeout 40 $SSH "squeue -u shenoy.am -h -o %j" 2>/dev/null | grep -c "cofold_coact" || echo 0)
say "coact job in queue: ${INQ}"

# 4. resubmit if not running/pending, not done, and not stopped
if [ -f "$STOP" ]; then
  say "STOP marker present -> no resubmit"
elif [ "${INQ:-0}" -eq 0 ] && [ "${DONE:-0}" -lt 4139 ]; then
  R=$(timeout 40 $SSH "cd $RD && sbatch cofold_coact_train.sbatch" 2>/dev/null)
  say "RESUBMITTED: $R"
elif [ "${DONE:-0}" -ge 4139 ]; then
  say "ALL 4139 DONE"; touch "$STOP"
fi

# 5. honest-gate eval
.venv/Scripts/python.exe scripts/nb1134_coact_train_eval.py 2>&1 | tee -a "$LOG"

# 6. early-kill if verdict is clearly in (redundant) at a representative n_done -> stop wasting GPU
.venv/Scripts/python.exe -c "
import json, os
f='data/processed/nb1134_coact_eval.json'
if os.path.exists(f):
    r=json.load(open(f)); nd=r.get('n_done',0); d=r.get('delta',0); v=r.get('verdict','')
    # clear redundancy at >=2500 compounds, OR confirmed help: either way the verdict is decided
    if nd>=2500 and d>-0.002:
        open(r'C:/pxr_work/coact_STOP','w').write(f'redundant: delta={d} at n={nd}')
        print('EARLY-KILL: ternary redundant (delta %+.4f at n=%d) -> STOP marker set'%(d,nd))
    elif nd>=2500 and d<=-0.005:
        print('CONFIRMED HELP: delta %+.4f at n=%d -> let it finish, then deploy'%(d,nd))
" 2>&1 | tee -a "$LOG"

# 7. if STOP just set for redundancy, cancel the running job to free the A100
if [ -f "$STOP" ] && grep -q redundant "$STOP" 2>/dev/null; then
  timeout 40 $SSH "scancel -n cofold_coact_train.sbatch -u shenoy.am" 2>/dev/null && say "scancelled redundant coact job (good citizen)"
fi
say "=== tick done ==="
