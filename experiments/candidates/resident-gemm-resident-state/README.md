# Resident boundary state

Keep W0 resident across frames after its complete P-step circulation restores canonical order. Load it from W_CANON only for frame zero, and remove redundant X1/W1 zeroing because full even-sized receives overwrite those scratch buffers before use. Preserve all input-frame copies, arithmetic, communication, shift resets, output stores and complete working-W0 snapshots. The full per-frame exact-state oracle must falsify any faulty residency assumption.

The contract explicitly allows removing a reset or copy only if every output and boundary state still passes. This is an unmeasured candidate. It does not reuse output answers or omit any GEMM frame/K term. W0 is loaded during the first timed frame, and all four weight snapshots remain inside the timer.

For the registered shapes, Mt*Kt and Kt*Nt are even; the communication DSDs receive every scratch element, and existing completion barriers precede pointer swaps. Compare all three shapes before a cross-shape claim.

Reject any six-case output, canonical input or working-W0 history mismatch, partial receive, changed arithmetic/timer/repeat/PE contract or shutdown failure. Retain only with >=1.05x matched cycle gain, <=1.10x program-memory ratio and fresh paired confirmation. Do not infer a suite-wide gain without every profile and worst-cell guards.
