# Evidence and rejected hypotheses

This is a curated numerical account. Unreviewed working notes, capture arrays, weights and personal trials were not
copied into the release. Historical findings and new release checks are deliberately identified separately.

## Frontend: the 828-frame claim

Target: Echo Dot 3 firmware 6574.1, ARMv7 Pryon/NemoRT, library SHA-256
`65df801e37ad0bbdfaf8922174bceb084fae6ba71c5076b1e23a000523f97027`.
Input: 16 kHz signed little-endian PCM, 400-sample windows, 160-sample hop, 512-point real FFT, 64 mel bins spanning
80–7200 Hz. A retained debugger capture has 138 complete six-frame conv1 inputs: 828 evaluated frames.

The first HybridConv uses im2col shape `360 × 35` (`6 × 60 × 7 × 5`). Checking overlapping frequency patches and taking
the newest history row recovers its 64-bin normalized features. The reconstruction processes the same PCM, applies the
original graph's own mean/inverse-standard-deviation tensors, then compares float32 bit patterns at that boundary.
The current release reanalysis gets **828/828 frames, 52,992/52,992 values identical, max error 0**.

The comparison requires privately supplied PCM, captures and normalization tensors. None are bundled. This was a fresh
read-only reanalysis of retained observations, not a new execution/capture of the stock library. Undoing normalization
is lossy: comparing raw LFBE against inversely recovered features instead gave 726 exact frames and max error
1.90735e-6. That is a different comparison boundary, not evidence against the normalized-boundary result.
`compare_frontend_capture.py` implements the explicit comparison without writing model/capture data.

Historical supporting stage checks: 358/358 FFT frames identical; 26,560/26,560 msun logf outputs identical; scalar mel
accumulation orders gave only 65–85% exact results until the measured OpenBLAS ARMv7 accumulation order was used.

## Streaming inference

Historical end-to-end runs evaluated 133 inference cycles: LSTM X max error 1.07e-6, and after cycle 30 classifier
input max error below 2.4e-6 with wake softmax below 1.8e-7. Conv2–7 kernel checks on captured stock im2col inputs
were bit-exact after correcting the quantized epilogue. These are distinct gates: a kernel fed stock inputs matching
does not mean every independently generated intermediate matches on every stream.

The initial decoder comparison matched 12/14 event scores to six digits, with two within .003. Later diagnosis of
one custom-head negative found a much larger score discrepancy: the vendor formed float32 affine BatchNorm
coefficients and rounded the affine expression once. Matching that behavior removed the known Conv2 cycle-42 flip
and reduced all callbacks of that diagnostic to within 4e-7, at identical event samples. This is finite diagnostic
evidence, not a bound on all phrases. Speech-boundary sample estimates remain incompletely derived.

## Training and numerical reproducibility

A floating classifier that looks good before export can change substantially under int8 conversion. The independent
forward path therefore quantizes during optimization, using straight-through gradients and recording-level temporal
windows. Hard maximum pooling of negative windows avoids diluting one false accept in a long negative recording.

Training loss improved, but a small recording-grouped study still had 4/12 validation errors, tying the float baseline.
Preceding dense bias/matrix adaptation did not improve that validation count. Acoustic augmentation and training-mean
export-bias compensation also failed their held-back checks. These are unsuccessful experiments, not discarded
benchmarks to conceal. Previously inspected held-out recordings became diagnostics; future quality claims require
fresh untouched recordings and speakers.

One false accept also occurred in stock Pryon with the same candidate, and already exceeded threshold in floating
head inference. Quantization or a reconstruction bug was not required to produce it. The evidence does not establish
that the frozen encoder can never separate a particular phrase.

Host/device optimizers initially emitted different models from the same inputs. glibc and Echo bionic exp/log/log1p
differed by one ulp, amplified through optimization and quantization. Historical device/QEMU checks agreed on
240,006 inputs. Porting the corresponding Sun/msun operation order made a retained native candidate byte-identical
across host/QEMU/device. The public C regression pins **64-bit FNV-1a**, not SHA-256, at `96594667583f1fd3` over its
specified deterministic sequence. It is not a trained-model hash or proof for every possible input.

## Physical work and what it does not establish

The Echo-side feature records through the device capture path, extracts stock encoder activations, fits the head
locally at low scheduling priority, checks held-back clips, packages a separate model and confirms root installation
before selection. An earlier web enrollment completed on the Dot 3; its small trial is feasibility evidence only.

Subsequent real-device tests used an isolated model/state tree: real root copy, startup checksum confirmation,
stock Pryon initialization/audio processing, missing/mismatched result recovery, and clearing refused during an
actual delayed root copy. The active application/model was preserved. During this publication pass no active Echo
configuration was changed; additional checks are recorded separately in VALIDATION.md.

A fresh full microphone enrollment after the recovery revision, broad false-accept/miss benchmarks, multiple users,
other Echo models, and robust continuous operation are not established. This is experimental engineering research,
not production-ready arbitrary-phrase recognition.


## Publication follow-up: making the Python math oracle portable

The initial hosted run passed, but a documentation-only follow-up failed the Python math-output SHA assertion while
the independent C FNV oracle still passed. A diagnostic branch pinned the original argument-array SHA and confirmed
that hosted runners generated different arguments **before any reconstructed math function was called**. The old
argument generator used NumPy floating-point exp/power operations; a fixed random seed alone was not a portable
bit-pattern contract.

The Python regression now constructs 240,006 IEEE-754 arguments using only integer xorshift operations and explicit
bit patterns. It pins input SHA-256 `8de28825c20549bfb3107263dc8ad64d420884aef299ed2e00fe8e62986ab97b`.
A new math-only probe ran these exact bytes through the actual Dot 3 libm as the daemon user; its output and the
unchanged reconstruction both gave SHA-256 `f29108dc1560898fc4133c2756ca084aacc782292fd26e57a82a12c7e085a269`
after canonicalizing NaNs. Protected files were unchanged, services were not restarted and temporary files were
removed. Only synthetic argument/output hashes are recorded here, not device identifiers or capture files.
The original C FNV oracle remains unchanged. No inference/training implementation or acceptance tolerance was altered.
