# Architecture and source map

The original ONNX has 7 HybridConv, 2 HybridLSTM and 4 HybridGemm operators. Observing the wake branch's final Gemm
identified a `[3,100]` quantized classifier acting on 25 independent 100-dimensional representations each cycle.
The other branch supports speech/end-point metadata; keyword boundary timing is not completely reconstructed.

| Component | Source | Important contract |
|---|---|---|
| DSP | `pryon_lfbe.c`, `pryon_runtime_analysis.py` | DC removal, preemphasis .96875, Povey window, Ne10-order real FFT, float mel weights, ARMv7 dot order, msun logf |
| Quantized network | `pryon_quant_model.py` | Per-row activation quantization, weight scales/zero-points, conv bias, affine BatchNorm rounding |
| LSTM and streaming | `pryon_lstm_analysis.py`, `pryon_native_stream.py` | Shared X broadcast to 25 state lanes; reset lane `t % 25`; six LFBE frames per cycle |
| Classifier and decoder | `pryon_decoder.py`, `pryon_inference.py` | Three-way softmax, ages 9–23, width-three means, youngest eight lanes trimmed, measured accept/near-miss and lock rules |
| Training | `train_pryon_streaming.py`, `pryon_qat.py` | Complete recordings supervise temporal windows; float initialization, exact int8-forward/STE optimization; training-only checkpoint choice |
| Adaptation studies | `pryon_dense_adapt.py` | Constrained preceding-dense bias/matrix experiments; source model and other tensors must remain unchanged |
| Original graph export | `pryon_head_quantize.py`, `pryon_training_io.py` | Deep-copy replacement, integer sums/scale/zero-point coherence; guards against non-target modifications |
| Native training | `native/pryon_train.c`, `native/pryon_qat.c` | Streaming recording-level objective and portable Echo-compatible math |
| Capture tools | `pryon_debug_capture.py`, `pryon_fft_capture.py` | Read-only QEMU/GDB observation, pinned executable/library hashes; user-supplied external Echo checkout |
| Float translation | `convert_pryon_native.py` | Earlier diagnostic hypothesis, optional LSTM broadcast; successful graph loading is not numerical parity |

Constant-range activation rows use a documented float fallback; that fallback is not a universally measured vendor
rule. Initialization can contain nonfinite values; training excludes the first 30 cycles. Resetting the entire encoder
for every short chunk would change its recurrent/history semantics. File-enrollment padding is a clip protocol and
must not be mistaken for continuous-stream reset behavior.

The standalone release does not duplicate the process-patching native extractor. Its reviewed implementation remains
in the Echo fork; this export retains the read-only debugger capture path and portable native trainer.
