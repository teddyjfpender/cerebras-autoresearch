# Distributed resident SwiGLU sequence

Derived from pinned WaferLLM MeshGEMM. See LICENSE.upstream. Three complete
FP16 projections, FP32 SiLU/product/residual, every intermediate and full boundary
state, four frames and one host launch. See docs/benchmarks/resident-swiglu.md.
