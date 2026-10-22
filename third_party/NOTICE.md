# Source attribution

- Cerebras SDK Examples: Apache License 2.0, pinned by `config/sources.lock.json`.
  License: `licenses/sdk-examples-LICENSE`. Sources are fetched into ignored `.deps/`.
- MeshInfra/WaferLLM: Apache License 2.0, pinned by the same lock file.
  License: `licenses/waferllm-LICENSE`. The checked-in GEMV candidate derives from
  this source; its main kernel has been modified to unroll the local FMA map.
  Modified support files carry notices for trailing whitespace cleanup.
- The dense host driver uses WaferLLM's tiling/shuffle interfaces. Its independent
  FP64 oracle, adversarial cases, local timestamp decoding, and receipts are local
  benchmark additions.

The Cerebras SDK container is separately licensed and is not redistributed.
No private SDK download URL, model weights, or hosted API credential is included.
