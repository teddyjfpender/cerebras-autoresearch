# bf16-swiglu-two-buffer-tail4

Remove the separate FP32 row accumulator: accumulate expansions in res_tile, transpose via dead W1 after all transfers complete, and copy back before protected BF16 stage stores. For Mt=1 the two layouts coincide. Vector activation reuses res_tile for polynomial output and W1 for squares, then FP32 U; mixed BF16 G*0.5 with FP32 FMA restores the linear SiLU term without another tile. Guard both uses by 2*Mt<=Ht. This trades a timed extra pack pass for lower program memory, preserving every term, precision boundary, frame and state store.

Reject skipped/reordered projection terms, BF16 accumulation, missing stage quantization, stale logical reads, numerical/state/shutdown failures, protected-control changes, speedup below 1.05x or memory above 1.10x. Require six families, fresh matched seed and complete suite guards for cross-shape claims. Reject any alias used while communication is live or any scratch extent exceeding W1; compare complete working-state histories at both seeds. Reject any transpose-before-transfer completion, scratch extent beyond W1, missing copy-back, incorrect identity-layout case or missing original G in the mixed FMA. Require complete four-shape full/fresh memory and worst-cell gates before widening the claim.

The scratch capacity is checked in bytes: `4*Mt*Ft <= 2*Ht*Ft`, hence
`2*Mt <= Ht`. W1 is aligned to four bytes and rebound at runtime.

| Phase | `res_tile` | W1 receive buffer |
| --- | --- | --- |
| Gate/up projection | FP32 row accumulation | Active weight send/receive buffer |
| After the final transfer callback | Complete row result; then column result copied back | Full FP32 transpose destination |
| Activation setup | Converted G, then polynomial output | Clamped G squared |
| Activation finish | `square*P + 0.5*G` via FP32 mixed FMA, then product | Squared G is dead; replaced by converted U |
| Down projection | Fresh FP32 output accumulation | Entire logical buffer overwritten by weight reception before use |

The last next-step callback is gated by both completed send/receive paths.
Packing and activation run after that boundary. Protected stage stores retain
BF16 G/U/V/D/Y and every canonical/working weight snapshot. A single-row tile
already has identical row/column ordering, so its transpose is the identity.
This source is a follow-up hypothesis after observing the held-out baselines;
its claims require new full/fresh four-cell qualification.
