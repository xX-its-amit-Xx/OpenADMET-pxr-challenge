#!/bin/bash
# Pre-download the full boltz-2 cache on the LOGIN node (compute nodes can't reach the HF CDN).
cd /scratch/$USER/boltz_pxr/boltz_cache || exit 1
B1=https://huggingface.co/boltz-community/boltz-1/resolve/main
B2=https://huggingface.co/boltz-community/boltz-2/resolve/main
echo "START $(date)"
[ -f ccd.pkl ]          || wget -q $B1/ccd.pkl
echo "CCD_DONE $(date)  size=$(stat -c %s ccd.pkl 2>/dev/null)"
[ -f boltz2_conf.ckpt ] || wget -q $B2/boltz2_conf.ckpt
echo "CONF_DONE $(date) size=$(stat -c %s boltz2_conf.ckpt 2>/dev/null)"
[ -d mols ]             || { wget -q $B2/mols.tar && tar xf mols.tar && rm -f mols.tar; }
echo "MOLS_DONE $(date) n=$(ls mols 2>/dev/null | wc -l)"
[ -f boltz2_aff.ckpt ]  || wget -q $B2/boltz2_aff.ckpt
echo "AFF_DONE $(date)  size=$(stat -c %s boltz2_aff.ckpt 2>/dev/null)"
echo "ALL_DONE $(date)"
ls -la
