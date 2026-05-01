# Refinement Theory Experiment Summary

## 1. Known Bayes-gap / regret-slope calibration

| delta | theory gap | repair/refined slope | Bayes-oracle slope | DGM U accepts | DGM noise accepts |
|---:|---:|---:|---:|---:|---:|
| 0.0 | 0.0000 | -0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.00 | 0.00 |
| 0.1 | 0.0201 | 0.0206 +/- 0.0013 | 0.0206 +/- 0.0013 | 3.88 | 0.00 |
| 0.2 | 0.0823 | 0.0824 +/- 0.0012 | 0.0824 +/- 0.0012 | 4.00 | 0.00 |
| 0.3 | 0.1927 | 0.1928 +/- 0.0041 | 0.1929 +/- 0.0041 | 4.00 | 0.00 |
| 0.4 | 0.3681 | 0.3682 +/- 0.0037 | 0.3682 +/- 0.0038 | 4.00 | 0.00 |

## 2. Guarded refine-or-repair with spurious distinctions

| scoring m | true recall | false refine | spurious accept | repair accuracy |
|---:|---:|---:|---:|---:|
| 32 | 0.956 | 0.001 | 0.001 | 0.999 |
| 64 | 0.991 | 0.000 | 0.000 | 1.000 |
| 128 | 1.000 | 0.000 | 0.000 | 1.000 |
| 256 | 1.000 | 0.000 | 0.000 | 1.000 |
| 512 | 1.000 | 0.000 | 0.000 | 1.000 |

## 3. Contextual XOR graph-memory ablation

| model | preq acc | preq NLL | test acc | concepts | edges | context edges | spurious edges | cand. recall | purity | ARI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full DGM refine+repair+edges | 0.959 +/- 0.002 | 0.073 | 0.998 | 8.0 | 4.0 | 4.0 | 0.0 | 1.000 | 0.997 | 0.994 |
| DGM repair-only | 0.503 +/- 0.007 | 0.696 | 0.502 | 4.0 | 0.0 | 0.0 | 0.0 | 0.000 | 0.514 | 0.600 |
| DGM refine-only | 0.501 +/- 0.007 | 0.693 | 0.496 | 8.0 | 4.0 | 4.0 | 0.0 | 1.000 | 0.997 | 0.994 |
| No-edge centroid | 0.504 +/- 0.004 | 0.698 | 0.502 | 8.0 | -- | -- | -- | -- | 0.155 | 0.002 |
| kNN/cache same budget | 0.497 +/- 0.006 | 0.785 | 0.499 | 8.0 | -- | -- | -- | -- | -- | -- |
| kNN/cache large budget | 0.494 +/- 0.005 | 0.788 | 0.498 | 64.0 | -- | -- | -- | -- | -- | -- |
