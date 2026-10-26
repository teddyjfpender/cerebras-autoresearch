# Resident SwiGLU sequence contract v1

Status: both initial cells passed full six-case baseline qualification and numerical/core replay. See the [qualification evidence](../../results/reports/2026-10-23/resident-swiglu/qualification.json).

## Objective and scope

Time a complete distributed feed-forward chain on a fixed 4x4 compute grid,
with one host launch for four distinct resident input batches and three shared
resident weight matrices. This extends the measured resident GEMMs to real
operator composition: gate projection, up projection, SiLU/product, down
projection, residual addition, and full output/state evidence.

This benchmark omits RMSNorm and attention. It uses scaled FP16 projections and
hidden storage with FP32 nonlinear/residual computation. It does not implement
Qwen's native BF16/hybrid decoder and cannot establish model or hardware TPS.

Qualified initial cells: M=16, H=16, F=32, T=4 and M=16, H=80, F=272, T=4.
Both use exactly 16 compute PEs. The larger cell preserves the FFN aspect ratio
already used by the qualified GEMM profiles. All dimensions are divisible by P=4.

## Mathematical and precision contract

Inputs X[t,M,H], Wg[H,F], Wu[H,F], Wd[F,H] are immutable quantized FP16 arrays.
For every t=0..3:

1. G = X[t] @ Wg, using all H FP16 FMA terms per output.
2. U = X[t] @ Wu, using all H FP16 FMA terms per output.
3. V = round16(SiLU32(float32(G)) * float32(U)).
4. D = V @ Wd, using all F FP16 FMA terms per output.
5. Y = float32(D) + float32(X[t]), stored in FP32.

SiLU(x)=x/(1+exp(-x)). Use its overflow-safe equivalent for negative x in the
reference implementation. Every G/U/V/D/Y element is saved for every frame.
Canonical input/weight bits must survive unchanged. The complete working-W0
tile after each final down projection must equal the canonically packed Wd
slice exactly. G/U working buffers are internal scratch; their complete matrix
outputs, and all three immutable canonical weights, are independently checked.
This is a distinct state contract from the single-GEMM sequence.

## Fixed numerical bounds, before device bringup

Let u16=2^-11, u32=2^-24 and q16=2^-24 (FP16 underflow allowance).
For a projection with inner dimension K, l=K+P and gamma=l*u16/(1-l*u16).
Require l*u16<1. For quantized operands A,B, bound every output by
`gamma*(abs(A) @ abs(B)) + l*q16`, relative to FP64 A@B.

Check G and U against immutable X/W inputs. Check V against FP64 SiLU of the
verified actual FP16 G/U operands, with local allowance
`4*u16*abs(SiLU(G)*U) + q16`. This fixed FP16-stage accuracy envelope includes
FP32 nonlinearity evaluation and FP16 hidden quantization; it is not adjusted
from observed device errors. Check D against FP64 actual-V @ Wd with the same
linear bound, and Y against actual-D + immutable-X with allowance
`4*u32*(abs(D)+abs(X)) + 2^-149`.

Also require a full-chain bound relative to the ideal FP64 chain of immutable
quantized inputs. For ideal gate/up G*,U* and their immutable-input bounds Eg,Eu,
use SiLU's global Lipschitz bound 1.5 and |SiLU(x)|<=|x|:

`Ev = 1.5*abs(U*)*Eg + (abs(G*)+Eg)*Eu
      + 4*u16*(abs(G*)+Eg)*(abs(U*)+Eu) + q16`.

(The derivative bound follows from |x|*sigmoid'(x)<=1/e<0.5.) For ideal V*,D*,

`Ed = Ev @ abs(Wd) + gammaF*((abs(V*)+Ev) @ abs(Wd)) + (F+P)*q16`,
`Ey = Ed + 4*u32*(abs(D*)+Ed+abs(X)) + 2^-149`.

All actual arrays must be finite, have the declared shapes and precision, and
pass both local-stage and ideal-chain checks. Host inputs use the six existing
families (positive, signed, zero, cancellation, one-hot, mixed scales), with
independently supplied frames and bounded weight scales to avoid overflow.
Changing workload, local tolerances, ideal propagation or state coverage needs
a documented new contract and fresh baselines. Numerical failures first require
implementation diagnosis; they are not grounds to loosen these bounds.

## Timing, storage, and metrics

A single whole-sequence timer includes all frame/stage setup, canonical working
copies, communication, all three GEMMs, nonlinear/product/residual work, every
intermediate/output store and every boundary W0 snapshot. No host reload or
launch is permitted between stages or frames. Initial H2D and post-stop core
inspection remain outside. Zero warmups, one sequence, one simulator thread and
a fresh process per numerical family; reject every shutdown failure.

Use `max_local_cycles`, program bytes/PE, PE-cycle spread, numerical bound
fractions, and declared projection arithmetic operations/cycle. The work count
`6*T*M*H*F` covers the three projections; nonlinear and residual work remains in
the timer and is explicitly excluded from that numerator. Report cycles/frame
and amortized cycles/batch row with their meanings stated.

A conservative first implementation may communicate padded X scratch sized to
max(H,F), retaining every logical term. Subsequent candidates may remove padded
movement or redundant scratch initialization when equivalent work and the full
output/state contract still hold. Timers, stage/frame controls and evidence
collection must be protected against candidate edits.

## Gates

Before registration: independent layout/packing tests, stage corruption tests,
canonical signed-zero checks, full-chain error propagation tests, and source
review. After registration: full six-case baselines on the small cell before
the scaled cell, raw numerical/core/timestamp replay, artifact/TSV verification.
Every V2 bringup failure remains in the journal. Candidate retention requires
>=1.05x matched cycle speedup, <=1.10x program-memory ratio, all six families
and a fresh paired seed. Cross-cell claims require the full suite/worst-cell guard.
