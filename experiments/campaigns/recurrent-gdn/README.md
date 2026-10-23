# Resident recurrent gated delta-rule plans

These plans require the separately staged recurrent-GDN evaluator to be registered
first. They currently have no device execution evidence. Qualify K16/V16 before
K32/V16; both keep four resident tokens, 16 key heads and three value heads per key
head on 16 PEs, BF16 Q/K/V and outputs, and complete FP32 state history.

Initial seed is 20261035; retained candidates require full validation and matched
fresh seed 20261036. This is the recurrent operator after projected/conv QKV and
precomputed decay/beta inputs; complete-model performance remains unmeasured.
