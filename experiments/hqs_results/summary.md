# HQS CIFAR-100 Hierarchy Query Stream

variant=balanced_mask, eta=0.02, steps=60000, repeats=5, proposal=64, score=64

| Method | Held-out NLL | Held-out acc. | Atoms/cells | Edges | Edge F1 | True edges | False edges |
|---|---:|---:|---:|---:|---:|---:|---:|
| Repair-only | 0.694 | 0.502 | 100.0 | 0.0 | -- | -- | -- |
| Frequency-only edge | 0.646 | 0.540 | 200.0 | 100.0 | 0.080 | 8.0 | 92.0 |
| Additive logistic | 0.674 | 0.581 | 121.0 | 0.0 | -- | -- | -- |
| Crossed logistic | 0.117 | 0.980 | 2001.0 | 0.0 | -- | -- | -- |
| Budgeted cache M=200 | 0.699 | 0.497 | 200.0 | 0.0 | -- | -- | -- |
| Budgeted cache M=500 | 0.695 | 0.499 | 500.0 | 0.0 | -- | -- | -- |
| Budgeted cache M=2000 | 0.693 | 0.499 | 2000.0 | 0.0 | -- | -- | -- |
| Per-fine local stump | 0.099 | 0.980 | 200.0 | 100.0 | 1.000 | 100.0 | 0.0 |
| DGM-HQS | 0.099 | 0.980 | 200.0 | 100.0 | 1.000 | 100.0 | 0.0 |
| Oracle refined | 0.099 | 0.980 | 200.0 | 100.0 | -- | -- | -- |
