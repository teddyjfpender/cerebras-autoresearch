# Native FP32 recurrent refinement

Immediately accumulate the BF16-output readout after each FP32 state row is updated. Preserve ascending K order and every full state store, while removing one separate row-walk loop.

The starting reference already uses FP32 vector state arithmetic. Every token,
full state, intermediate, canonical input and protected timer/history function
is preserved. See candidate.json for the preregistered falsification gates.
