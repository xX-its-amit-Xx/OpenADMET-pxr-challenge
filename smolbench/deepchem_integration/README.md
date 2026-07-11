# DeepChem integration

A DeepChem-native version of smolbench's honest benchmark, using DeepChem's own
featurizers (`CircularFingerprint`, `RDKitDescriptors`, `MACCSKeysFingerprint`),
splitters (`ScaffoldSplitter`, `ButinaSplitter`) and `NumpyDataset` — no external deps.

**Upstream PR:** https://github.com/deepchem/deepchem/pull/5054
(adds `deepchem/utils/benchmark_utils.py` + tests)

These files mirror what was submitted to DeepChem, kept here for reference.
