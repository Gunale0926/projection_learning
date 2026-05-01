# HQS CIFAR-100 Hierarchy Query Stream

eta=0.02, steps=60000, repeats=5, proposal=64, score=64

| Method | Held-out NLL | Held-out acc. | Atoms/cells | Edges | Edge F1 | True edges | False edges |
|---|---:|---:|---:|---:|---:|---:|---:|
| Repair-only | 0.694 | 0.498 | 100.0 | 0.0 | -- | -- | -- |
| Additive logistic | 0.696 | 0.508 | 121.0 | 0.0 | -- | -- | -- |
| Cross-product memory | 0.115 | 0.980 | 2000.0 | 0.0 | -- | -- | -- |
| Crossed logistic | 0.101 | 0.980 | 2001.0 | 0.0 | -- | -- | -- |
| Budgeted cache M=200 | 0.590 | 0.732 | 200.0 | 0.0 | -- | -- | -- |
| Budgeted cache M=500 | 0.507 | 0.704 | 500.0 | 0.0 | -- | -- | -- |
| Budgeted cache M=2000 | 0.366 | 0.862 | 2000.0 | 0.0 | -- | -- | -- |
| DGM-HQS | 0.099 | 0.980 | 200.0 | 100.0 | 1.000 | 100.0 | 0.0 |
| Oracle refined | 0.099 | 0.980 | 200.0 | 100.0 | -- | -- | -- |
