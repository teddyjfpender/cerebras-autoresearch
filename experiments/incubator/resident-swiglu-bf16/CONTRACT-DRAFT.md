# Proposed BF16 storage / FP32 accumulator contract

Unregistered design; no SDK capability, execution or speed result is claimed.
Freeze the reviewed final specification before device bringup and registration.

Let ub=2^-8, u32=2^-24, qb=2^-133 and q32=2^-149. Quantize immutable X/W inputs
to BF16 round-to-nearest-even and retain exact uint16 words. Decode to FP64 for
independent arithmetic. Require finite inputs, intermediates and outputs; use
bounded six-family inputs with no overflow. Inputs and every canonical weight
must survive bit-exact, including signed zero. Preserve every frame's final
working down-weight tile. Every projection executes all requested products.

For projection inner dimension K, l=K+P and gamma32=l*u32/(1-l*u32), define

```
R = FP64(A) @ FP64(B)
Eacc = gamma32 * (abs(A) @ abs(B)) + l*q32
Eprojection = Eacc + ub*(abs(R)+Eacc) + qb
```

This bounds FP32 FMA accumulation followed by BF16 output quantization. It does
not permit BF16 accumulation or silently inherit the old FP16 numerical contract.

For actual verified BF16 gate/up operands Ga,Ua, verify V against
`SiLU64(Ga)*Ua` with fixed envelope `4*ub*abs(SiLU64(Ga)*Ua)+qb`. For actual V,
verify D against its FP64 product with Wd and the projection bound above.
For actual D and immutable X, set RY=D+X, Eadd=4*u32*(abs(D)+abs(X))+q32;
verify BF16 Y against RY with `Eadd+ub*(abs(RY)+Eadd)+qb`.

Also enforce an independent ideal full-chain output from immutable inputs. For
ideal G*,U* with their projection bounds Eg,Eu:

```
V* = SiLU64(G*)*U*
Ev = 1.5*abs(U*)*Eg + (abs(G*)+Eg)*Eu
     + 4*ub*(abs(G*)+Eg)*(abs(U*)+Eu) + qb
D* = V* @ Wd
Ein = Ev @ abs(Wd)
EaccD = gamma32_F*((abs(V*)+Ev) @ abs(Wd)) + (F+P)*q32
Ed = Ein + EaccD + ub*(abs(D*)+Ein+EaccD) + qb
Y* = D* + X
EaddY = 4*u32*(abs(D*)+Ed+abs(X)) + q32
Ey = Ed + EaddY + ub*(abs(D*)+Ed+abs(X)+EaddY) + qb
```

Require the local checks and `abs(Y-Y*)<=Ey`, complete declared shapes, raw BF16
bit consistency, and all canonical/boundary state. Tests must reject low-bit
numeric drift, signed-zero changes, stage/output/cache corruption, wrong precision
and missing frames, even when outer artifact hashes are recomputed.

Timing includes every stage/frame, conversions, arithmetic, movement, nonlinear
work, full intermediate/output and state stores, with no host reload/launch
between stages. One thread, fresh process/case, clean shutdown, no warmup reuse.
Report total max local cycles, cycles/frame, amortized cycles/batch row, declared
projection operations/cycle, PE spread, numerical errors and program bytes/PE.
Profile and suite retention gates remain 1.05x, 1.10x memory, fresh full pairs,
complete suite coverage and fixed worst-cell guards. Existing FP16 scores stay
separate. The installed SDK's exact BF16 mode/rounding semantics must be tested.
