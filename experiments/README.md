# Experiments

The paper uses two empirical tracks, both backed by the PoST framework.

1. MQAR state-equalized grid:
   `../PoST_dev/zoology/zoology/experiments/ca_mqar.py`

2. CALM language-model pretraining:
   `../PoST_dev/trainer.py --arch calm`

Final same-corpus perplexity for LM checkpoints is evaluated with:
`experiments/eval_final_ppl.py`

Older CIFAR test-time adaptation, tiny language-model scripts, readout-only
language-model memory, and synthetic refinement diagnostics are not part of the
current paper because they test different claims.
