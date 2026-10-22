# GEMM map-over-K, supported base pointer

Unmeasured revision of `gemm-row-major-map-k`. The original attempt failed
compilation because `&ptr_X_send[row]` has a single-element pointer type; the
compiler requires an array or many-item base for this descriptor operation.
Set the descriptor base to `ptr_X_send`, then apply the FP16 row offset.

The intended compute mapping, all reduction terms, communication, precision,
output transpose and protected timer/control functions are unchanged. Preserve
the original compile-failure receipt. Screen both FFN shapes before any full
challenge; no performance claim follows from this source correction.
