# Resident GEMM sequence

Status: all three registered profiles passed device compilation, six-case
validation and independent output/state/timestamp replay. See the
[qualification report](../../../results/reports/2026-10-23-reusable-gemm-qualification.json).
Candidate improvements still require matched full validation and fresh confirmation.

Derived from WaferLLM MeshGEMM at
`fd1c2daae37cd68706c03fc8009887ecee9900f8`; see `LICENSE.upstream`.
The program computes four resident FP16 input matrices against shared weights
on a fixed 4x4 compute grid. One host launch covers all four frames, timed working
buffer/counter restoration, computation/communication, every output store and
all working-weight snapshots. Canonical input arrays are checked after the run.

Profiles are in `benchmarks/manifests/dense/reusable-gemm.json`. The independent
FP64 oracle and exact state checks live in `benchmarks/drivers/reusable_gemm_oracle.py`.
The full-sequence cycle score and cycles/GEMM are separate from cold GEMM scores.
They do not establish hardware or model TPS.
