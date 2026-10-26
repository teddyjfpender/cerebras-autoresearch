# Staged BF16 resident FFN baseline

Unregistered precision contract: native BF16 storage and operands with FP32
projection accumulators, FP32 SiLU/product/residual, then BF16 stage/output
rounding. Every G/U/V/D/Y stage and all canonical/boundary weight state remain
checked. Derived from the independently qualified FP16 chain, with distinct
precision, oracle, compiler format flag and baselines required.
