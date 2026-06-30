@echo off
"C:\Program Files\Git\usr\bin\bash.exe" -c "cd /d/Users/ashenoy00000/.windsurf/OpenADMET-pxr-challenge && .venv/Scripts/python.exe scripts/nb1126_combinatorial_search.py --batch 80 >> C:/pxr_work/swarm_search.log 2>&1 && .venv/Scripts/python.exe scripts/nb1130_ensemble_check.py >> C:/pxr_work/swarm_search.log 2>&1"
