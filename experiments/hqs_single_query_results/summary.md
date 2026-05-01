# HQS CIFAR-100 Hierarchy Query Stream

variant=single_query, eta=0.02, steps=60000, repeats=5, proposal=64, score=64

| Method | Held-out NLL | Held-out acc. | Atoms/cells | Edges | Edge F1 | True edges | False edges |
|---|---:|---:|---:|---:|---:|---:|---:|
| Repair-only | 0.694 | 0.503 | 100.0 | 0.0 | -- | -- | -- |
| Frequency-only edge | 0.096 | 0.981 | 200.0 | 100.0 | 1.000 | 100.0 | 0.0 |
| Additive logistic | 0.697 | 0.508 | 121.0 | 0.0 | -- | -- | -- |
| Cross-product memory | 0.112 | 0.981 | 2000.0 | 0.0 | -- | -- | -- |
| Crossed logistic | 0.097 | 0.981 | 2001.0 | 0.0 | -- | -- | -- |
| Budgeted cache M=200 | 0.586 | 0.687 | 200.0 | 0.0 | -- | -- | -- |
| Budgeted cache M=500 | 0.497 | 0.706 | 500.0 | 0.0 | -- | -- | -- |
| Budgeted cache M=2000 | 0.367 | 0.862 | 2000.0 | 0.0 | -- | -- | -- |
| Per-fine local stump | 0.096 | 0.981 | 200.0 | 100.0 | 1.000 | 100.0 | 0.0 |
| DGM-HQS | 0.096 | 0.981 | 200.0 | 100.0 | 1.000 | 100.0 | 0.0 |
| Oracle refined | 0.095 | 0.981 | 200.0 | 100.0 | -- | -- | -- |
