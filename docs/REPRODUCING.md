# Reproducing the checks

## Public, firmware-free path

Tested with Python 3.12.8 on Linux x86_64. Install `requirements.lock.txt` in an isolated virtual environment, then:

```sh
make test PYTHON=.venv/bin/python
make demo PYTHON=.venv/bin/python
make build/pryon-train
```

`make test` runs the deterministic C math oracle and 32 synthetic Python tests. The suite includes a float64 frontend
reference, synthetic frontend golden vector, ONNX LSTM/broadcast contracts, affine BatchNorm regression, int8-forward
parity, gradient checks, temporal window construction, train/test ordering, original-model preservation and dense
adaptation constraints. The golden vector's provenance is this reconstruction, not an Amazon capture.

The native trainer reads the documented feature/manifest format; building it does not train or produce a useful
wake-word model by itself. The functioning microphone/web enrollment integration lives in the Echo fork.
For the NDK r21e cross-build, point `ECHO_SOURCE` at a local Echo checkout containing its compiler:

```sh
make cross ECHO_SOURCE="$ECHO_SOURCE"
```

No vendor model/library is needed for the trainer cross-build. Native training math is built with
`-ffp-contract=off`; substituting host libm or changing floating-point contraction can change trained outputs.

## Optional comparison against privately supplied genuine-runtime captures

This path is separate from the public synthetic suite. Supply your own model, matching raw PCM and read-only GDB
capture (`hits.json` and the corresponding `*-x.bin` files). These are neither distributed nor downloaded here.

```sh
.venv/bin/python compare_frontend_capture.py \
  --model private-assets/source.onnx \
  --capture private-assets/capture \
  --pcm private-assets/input.raw
```

The report compares normalized conv1 inputs directly. It does not execute Pryon, dump weights, copy captures into
the repository, or invert normalization. The reference report is `FRONTEND_RECHECK.json`; other firmware/builds are
not assumed compatible. The measured input layout and normalization nodes are checked for shape/overlap.

To collect fresh observations, use the source-only debugger tools with `--source "$ECHO_SOURCE"`. Their source checkout
must contain user-owned extracted firmware, `build/donut/pryon_test`, and caller-supplied PCM. Both scripts pin the
library SHA-256 and the harness SHA-256 before using offsets. QEMU and a PID namespace are required by this bionic
runtime. Do not remove identity gates or alter host security policy to bypass a namespace failure.

The verified harness recipe is in the Echo fork's `tools/pryon-native/README.md`: Android NDK r21e
(21.4.7075529; Clang 9.0.9, build 7019983), API 24. NDK archive SHA-256:
`ad7ce5467e18d40050dc51b8e7affc3e635c85bd8c59be62de32352328ed467e`.
Harness SHA-256: `107e8079ca73ac0c2d30d4cc1bb2cd62d78ffe8202bf64fa97cc643cdea23f84`, ARM main `0x1098`.
No proprietary executable is supplied. The tools have no firmware-write, service-restart or device-selection step.
