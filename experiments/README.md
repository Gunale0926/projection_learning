# Experiments

The paper keeps only two empirical tracks.

1. MQAR in the PoST framework:
   `../PoST_dev/zoology/zoology/experiments/cga_mqar.py`

2. TinyCGA-LM in this repository:
   `experiments/train_tiny_cga_lm.py`

3. Final same-corpus perplexity for LM checkpoints:
   `experiments/eval_final_ppl.py`

Older CIFAR test-time adaptation, readout-only language-model memory, and
synthetic refinement diagnostics were removed from the main experiment tree
because they test different claims.
