Update the README.md for this PXR challenge repo to reflect the current state of all models and ensemble results.

Steps:
1. Scan notebooks/ for any notebooks numbered higher than those currently documented in README.md (look at the "Models" and "Ensemble Progression" tables).
2. For each new notebook, extract the OOF RAE from its cell outputs using:
   ```
   python -c "
   import json, glob
   for nb_path in sorted(glob.glob('notebooks/*.ipynb')):
       nb = json.load(open(nb_path))
       for cell in nb['cells']:
           for o in cell.get('outputs', []):
               text = ''.join(o.get('text', o.get('data', {}).get('text/plain', [])))
               if 'OOF RAE' in text:
                   print(nb_path, text.strip()[:200])
   "
   ```
3. Check data/processed/ for any new oof_*.npy files not yet in the ensemble.
4. Check submissions/ for the latest grand_v*.csv to find the current best nested CV RAE.
5. Update README.md:
   - Add new rows to the "Models" table
   - Add new rows to the "Ensemble Progression" table
   - Update the "Final Ensemble Weights" table if a newer grand ensemble exists
   - Update the "Pipeline" file listing
   - Update the "Results" section with the new best RAE
6. Commit the updated README.md with message "Update README: add [model names] and grand vN (RAE X.XXXX)"

Do not change the structure or tone of the README — only add new information. Do not remove or alter existing rows unless they contain factual errors.