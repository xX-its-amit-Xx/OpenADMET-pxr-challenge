"""Protonation / tautomer-microstate thermodynamics feature block (ligand-only,
pure-SMILES, CPU-cheap) for the 4652-mol corpus (4139 train + 513 test).

Distinct axis vs deployed QM blocks (AIMNet2 electronic, MMFF strain conformational,
DFT-D4 dispersion/polarizability): the pH-7.4 *protomer* and titration behaviour.
Most PXR analogs are ionizable; the neutral-SMILES ECFP / RDKit descriptors in
`combined` are blind to which species actually exists (and binds/activates) at
physiological pH. dimorphite-dl (2.x, pKa-substructure enumerator) gives the
protonation microstates; we derive titration scalars per molecule:

  prot_q74          net formal charge of the dominant pH-7.4 protomer
  prot_qabs74       |charge| at pH 7.4
  prot_nstate_74    # distinct protomers at pH 7.4 (protonation AMBIGUITY near physio)
  prot_nstate_rng   # distinct protomers across pH 4..10 (titratable richness)
  prot_ncharge_rng  # distinct NET CHARGES across pH 4..10
  prot_qmin_rng     min net charge over pH 4..10 (acid character)
  prot_qmax_rng     max net charge over pH 4..10 (base character)
  prot_charge_shift q74 - neutral-input formal charge (info the naive SMILES MISSES)
  prot_frac_ion     fraction of sampled pH points where dominant species is charged

Resumable per-row append. Output: C:/pxr_work/protonation/prot_features.csv
"""
import os, sys, csv
import pandas as pd
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
from dimorphite_dl import protonate_smiles

CORPUS = "C:/pxr_work/xtb/corpus.csv"
OUTDIR = "C:/pxr_work/protonation"
OUT = f"{OUTDIR}/prot_features.csv"
PH_SCAN = [4.0, 5.0, 6.0, 7.0, 7.4, 8.0, 9.0, 10.0]
COLS = ["prot_q74", "prot_qabs74", "prot_nstate_74", "prot_nstate_rng",
        "prot_ncharge_rng", "prot_qmin_rng", "prot_qmax_rng",
        "prot_charge_shift", "prot_frac_ion"]


def fcharge(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return Chem.GetFormalCharge(m)


def states_at(smi, ph, precision):
    try:
        out = protonate_smiles(smi, ph_min=ph, ph_max=ph, precision=precision)
    except Exception:
        return []
    return [s for s in out if s]


def featurize(smi):
    neutral_q = fcharge(smi)
    if neutral_q is None:
        return None
    # pH 7.4 microstates (precision 1.0 -> ambiguity window ~+-1 pKa unit)
    st74 = states_at(smi, 7.4, 1.0)
    if not st74:
        st74 = [smi]
    q74_list = [fcharge(s) for s in st74]
    q74_list = [q for q in q74_list if q is not None]
    if not q74_list:
        q74_list = [neutral_q]
    # dominant = the charge state nearest to the median (representative); first is fine
    q74 = q74_list[0]
    nstate_74 = len(set(st74))
    # titration scan: representative (first) protomer per pH point, tight precision
    charges, smis = [], set()
    for ph in PH_SCAN:
        sts = states_at(smi, ph, 0.5)
        if not sts:
            continue
        smis.update(sts)
        qs = [fcharge(s) for s in sts]
        qs = [q for q in qs if q is not None]
        if qs:
            charges.append(qs[0])
    if not charges:
        charges = [neutral_q]
    nstate_rng = len(smis) if smis else 1
    distinct_q = sorted(set(charges))
    return {
        "prot_q74": q74,
        "prot_qabs74": abs(q74),
        "prot_nstate_74": nstate_74,
        "prot_nstate_rng": nstate_rng,
        "prot_ncharge_rng": len(distinct_q),
        "prot_qmin_rng": min(charges),
        "prot_qmax_rng": max(charges),
        "prot_charge_shift": q74 - neutral_q,
        "prot_frac_ion": sum(1 for q in charges if q != 0) / len(charges),
    }


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    corp = pd.read_csv(CORPUS)
    done = set()
    if os.path.exists(OUT):
        try:
            done = set(pd.read_csv(OUT)["name"].astype(str))
        except Exception:
            done = set()
    new = corp[~corp["name"].astype(str).isin(done)]
    print(f"corpus={len(corp)} done={len(done)} todo={len(new)}", flush=True)
    header = ["name", "src", "smiles"] + COLS
    write_header = not os.path.exists(OUT)
    f = open(OUT, "a", newline="")
    w = csv.writer(f)
    if write_header:
        w.writerow(header)
    n = 0
    for _, r in new.iterrows():
        feat = featurize(r["smiles"])
        if feat is None:
            feat = {c: 0 for c in COLS}
        w.writerow([r["name"], r["src"], r["smiles"]] + [feat[c] for c in COLS])
        n += 1
        if n % 250 == 0:
            f.flush()
            print(f"  {n}/{len(new)}", flush=True)
    f.flush(); f.close()
    print(f"DONE wrote {n} rows -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
