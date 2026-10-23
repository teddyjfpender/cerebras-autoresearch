# Fixed output descriptor accumulation

For Mt4, load four complete output descriptor pairs once per local matrix product, retain them across all K terms, and load each weight vector once per K. This removes repeated per-row output descriptor setup while preserving exact FP16 K-term order, timed transpose, full frame state and residency invariants. Shift-only banks4/5/6 are used after shift completion; in-flight two-hop banks2/3 stay untouched.

Reject any simulator/shutdown failure, omitted or reordered arithmetic term, six-case output/state failure or protected-control change. Require full fresh paired seeds, at least1.05x matched speedup and at most1.10x program memory. Cross-shape claims require every suite cell and its worst-cell guard.
