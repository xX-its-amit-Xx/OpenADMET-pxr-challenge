@echo off
REM Re-enables the paused PXR swarm tasks (fired once at Sun 1 PM), then deletes itself.
for %%T in (PXR_DataQueen PXR_ModelQueen PXR_Worker PXR_Worker2 PXR_Worker3 PXR_Worker4 PXR_Combinator PXR_Viz PXR_Search PXR_DailyDigest pxr_auto_submit_activity) do schtasks /change /tn "%%T" /ENABLE
schtasks /delete /tn "PXR_Reenable" /f
