# Recurrent reference decay accuracy repair

The original reference failed its existing FP32 exp bound on a canonical
positive-family input: max fraction1.10927739298, with every other check passing.
The failed receipt remains immutable and journaled. This workspace corrects the
reference math implementation using range reduction and degree-nine Horner,
with the same FP32 precision, all recurrent work, state stores and original bound.
It establishes no speedup against the invalid reference. A corrected canonical
source requires entirely new full baselines before optimization retention.
