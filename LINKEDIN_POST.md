Just wrapped the OpenADMET PXR Blind Challenge — predicting how 513 drug-like molecules activate the pregnane X receptor (PXR), the body's master xenobiotic sensor and a notorious driver of drug–drug interactions. ~1,800 modeling experiments later, here's what I'm proudest of, what humbled me, and what I'm giving back. 🧬

𝗧𝗵𝗲 𝟯 𝗺𝗼𝘀𝘁 𝗰𝗿𝗲𝗮𝘁𝗶𝘃𝗲 𝘀𝘄𝗶𝗻𝗴𝘀

🔬 𝗖𝗼-𝗳𝗼𝗹𝗱𝗶𝗻𝗴 𝘁𝗵𝗲 𝘁𝗮𝗿𝗴𝗲𝘁 𝘄𝗶𝘁𝗵 𝗲𝘃𝗲𝗿𝘆 𝗹𝗶𝗴𝗮𝗻𝗱. Instead of stopping at 2D fingerprints, I ran Boltz-2 to co-fold PXR with each molecule and pulled the protein×ligand *interaction* embedding. It was the ONLY feature that broke past the 2D ceiling — and remarkably, it detected activity cliffs (near-identical molecules with opposite activity) at AUC 0.84. Real target-aware structure carried signal nothing else did.

🧪 𝗧𝘂𝗿𝗻𝗶𝗻𝗴 𝗮 "𝘁𝗵𝗿𝗼𝘄𝗮𝘄𝗮𝘆" 𝘀𝗰𝗿𝗲𝗲𝗻 𝗶𝗻𝘁𝗼 𝘀𝗶𝗴𝗻𝗮𝗹. Buried in the data was a 21k-compound single-concentration functional screen everyone treats as noise. I distilled it into an orthogonal P(active) prior + an "inactive gate." This was the single lever that actually **transferred to the truly-blind set** (−0.019 RAE) while fancier corrections didn't.

📏 𝗥𝗲𝗳𝘂𝘀𝗶𝗻𝗴 𝘁𝗼 𝗹𝗶𝗲 𝘁𝗼 𝗺𝘆𝘀𝗲𝗹𝗳. Scaffold/leave-series-out CV throughout, adversarial train-vs-test checks, 15-seed stability bands. Random CV was ~0.1 RAE optimistic — a trap on novel chemistry.

𝗪𝗵𝗮𝘁 𝘀𝘂𝗿𝗽𝗿𝗶𝘀𝗲𝗱 𝗺𝗲 (𝘁𝗵𝗲 𝗽𝗼𝘀𝘁-𝗵𝗼𝗰 𝗿𝗲𝗰𝗸𝗼𝗻𝗶𝗻𝗴)

Now that the 260 blind labels are out, I scored everything honestly — and the biggest lesson stung:

😬 𝗠𝘆 𝗳𝗶𝗻𝗲𝗹𝘆-𝘁𝘂𝗻𝗲𝗱 𝗲𝗻𝘀𝗲𝗺𝗯𝗹𝗲 𝗼𝘃𝗲𝗿𝗳𝗶𝘁. The meta-stacker that looked BEST on validation (RAE 0.614) was the WORST component on the blind set (0.731) — and I'd given it 40% weight. My single most robust model alone would have scored ~0.63 vs my submitted 0.66, landing right at the leaderboard's statistical-tie cluster. On small, series-shifted data, trust one robust model over a beautiful stack.

🧗 𝗧𝗵𝗲 𝘄𝗮𝗹𝗹 𝗶𝘀 𝗿𝗲𝗮𝗹. My two worst-predicted compounds were called "active" by *every* signal I had — structural neighbors AND the functional screen said P(active) 0.92–0.99 — yet they measure inactive. True activity cliffs that no feature, 2D or 3D, resolves. They alone cost ~7% of my error.

🏆 𝗧𝗵𝗲 𝗰𝗲𝗹𝗲𝗯𝗿𝗮𝘁𝗲𝗱 "𝘄𝗶𝗻𝗻𝗲𝗿 𝘁𝗿𝗶𝗰𝗸" 𝗱𝗶𝗱𝗻'𝘁 𝘁𝗿𝗮𝗻𝘀𝗳𝗲𝗿. TabPFN-on-CheMeleon, run correctly, scored *worse* than a plain component on PXR and got absorbed to zero weight. Context matters more than hype.

𝗔𝗳𝘁𝗲𝗿 𝘁𝗵𝗲 𝗿𝗲𝘃𝗲𝗮𝗹: 𝟯 𝗺𝗼𝗼𝗻𝘀𝗵𝗼𝘁𝘀 𝗜 𝗳𝗶𝗻𝗮𝗹𝗹𝘆 𝗯𝘂𝗶𝗹𝘁

With the blind labels out, I built the three representation ideas I'd scoped but never finished — and scored them honestly on the truly-blind 260:

🧬 A 𝗵𝗶𝗲𝗿𝗮𝗿𝗰𝗵𝗶𝗰𝗮𝗹 𝗰𝘂𝗿𝗿𝗶𝗰𝘂𝗹𝘂𝗺 (pretrain broad Tox21 xeno-sensing → fine-tune on the nuclear-receptor family → fine-tune on PXR) genuinely worked: −0.048 RAE over training from scratch, and the *order mattered* — skipping the NR-family middle stage hurt. Biological hierarchy is real, transferable signal.
🧠 A 𝟯-𝗵𝗲𝗮𝗱 𝗻𝗲𝘁 (PXR potency + an assay-noise head + a dedicated activity-cliff head) — the auxiliary heads measurably regularized the shared representation.
🔗 A 𝗯𝗶𝗼𝗹𝗼𝗴𝗶𝗰𝗮𝗹 𝗳𝗶𝗻𝗴𝗲𝗿𝗽𝗿𝗶𝗻𝘁 reading activity across related receptors.

The punchline that ties the whole challenge together: 𝗲𝘃𝗲𝗿𝘆 𝗼𝗻𝗲 𝗽𝗿𝗼𝗱𝘂𝗰𝗲𝗱 𝗮 𝗿𝗲𝗮𝗹, 𝗰𝗼𝗿𝗿𝗲𝗰𝘁𝗹𝘆-𝘀𝗶𝗴𝗻𝗲𝗱 𝗲𝗳𝗳𝗲𝗰𝘁 — 𝘆𝗲𝘁 𝗻𝗼𝗻𝗲 𝗯𝗲𝗮𝘁 𝘁𝗵𝗲 𝗴𝗿𝗮𝗱𝗶𝗲𝗻𝘁-𝗯𝗼𝗼𝘀𝘁𝗲𝗱 𝗯𝗮𝘀𝗲𝗹𝗶𝗻𝗲. The cross-receptor tricks are capped by coverage (the blind compounds sit at just 0.28 Tanimoto to any public receptor panel), and the cliff head can't tell you which *direction* to correct. Same wall, hit from three new angles — and it's an information/coverage wall, not a modeling one.

𝗚𝗶𝘃𝗶𝗻𝗴 𝗶𝘁 𝗯𝗮𝗰𝗸

The hardest-won lesson — small chemical datasets quietly punish over-tuning — is now a tool. I packaged the honest-CV benchmarking harness as 𝘀𝗺𝗼𝗹𝗯𝗲𝗻𝗰𝗵: give it a train/test CSV and it searches featurizers × models × prep under scaffold CV, and *warns you* when random CV is lying to you.

📦 PyPI: https://pypi.org/project/smolbench/  (`pip install smolbench`)
🔗 Contributed a DeepChem-native version upstream — PR #5054 to the DeepChem team, whose splitters/featurizers made this whole campaign possible. 🙏
📊 Full post-hoc analysis (per-compound): https://github.com/xX-its-amit-Xx/OpenADMET-pxr-challenge/blob/main/POSTHOC_ANALYSIS.md
📝 Method + model report: https://xx-its-amit-xx.github.io/OpenADMET-pxr-challenge/
💻 Code: https://github.com/xX-its-amit-Xx/OpenADMET-pxr-challenge

Huge thank you to the 𝗢𝗽𝗲𝗻𝗔𝗗𝗠𝗘𝗧 team for running a genuinely blind, rigorously designed challenge with real experimental data. Blind benchmarks are how the field stays honest — and this one taught me more from where I was wrong than from where I was right. 🙏

#DrugDiscovery #MachineLearning #Cheminformatics #QSAR #OpenScience #ADMET #ComputationalChemistry #DeepChem #OpenADMET
