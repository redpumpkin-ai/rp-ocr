# Exploratory CPU trials

These reports preserve earlier FP32 margin/straight-line experiments and an
adaptive INT8 detector experiment. The runtime versions, profile implementations
and build settings differ between these stages. The `rust-fast` label in these
files refers to the implementation at that stage.

The margin and line routes did not establish a consistent speed advantage.
The INT8 detector introduced regressions on faded, low-resolution and table
examples. Its fallback variant preserved the development-set accuracy but
added latency. These paths were removed from the delivered engine.

The `int8-recognizer-only*` reports cover the full-alphabet INT8 reader with
original-resolution FP32 detection, before language specialization. The
`latin-head-validation.json` probe confirms identical transcripts on all 18
development images after pruning 6,906 classes to 692, before changing detector
resolution. These are accuracy development checks, not independent test data.

The current adaptive-resolution FP32 detector and Latin INT8 reader are evaluated in
[the optimization report](../OPTIMIZATION.md), with its raw repeated measurements.
