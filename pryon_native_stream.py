#!/usr/bin/env python3
"""Stream LFBE frames through the native float ONNX copy the way the stock
runtime does (measured in docs/EVIDENCE.md):

- 6 new LFBE frames per inference cycle, conv history buffers fed back;
- LSTM states fed back, and before cycle t state row t % 25 is zeroed
  (all rows at cycle 0) for both branches; one X row drives all 25 rows.

Float weights only: this is not bit-parity with the quantized kernels.
"""
import numpy as np
import onnxruntime as ort

FRAMES_PER_CYCLE = 6
STATES = {'input_h_ww': 'lstm_ww_1', 'input_c_ww': 'lstm_ww_1_1',
          'input_h_se': 'lstm_se_1', 'input_c_se': 'lstm_se_1_1'}


def session(path):
    options = ort.SessionOptions(); options.log_severity_level = 3
    return ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])


def stream(sess, frames, conv_init=np.nan):
    """Yield one output dict per full 6-frame cycle. conv_init: initial conv
    history value (the stock runtime's early X is non-finite)."""
    inputs = {i.name: i for i in sess.get_inputs()}
    buffers = [n for n in inputs if n.startswith('streaming_conv_buffer_input_')]
    feed = {n: np.full([1]+inputs[n].shape[1:], conv_init, np.float32) for n in buffers}
    feed.update({n: np.zeros((25, 100), np.float32) for n in STATES})
    names = [o.name for o in sess.get_outputs()]
    for t in range(len(frames)//FRAMES_PER_CYCLE):
        for n in STATES:
            feed[n][t % 25 if t else slice(None)] = 0
        chunk = frames[t*FRAMES_PER_CYCLE:(t+1)*FRAMES_PER_CYCLE]
        feed['input_feats'] = np.asarray(chunk, np.float32)[None, :, :, None]
        out = dict(zip(names, sess.run(None, feed)))
        for n in buffers:
            feed[n] = out[n.replace('_input_', '_output_')]
        for n, o in STATES.items():
            feed[n] = out[o].copy()
        yield out


def compare(native, audio, capture, warmup=30):
    """Native float stream vs a stock --gemm --classifier-only capture."""
    import json, onnx, tempfile
    from pathlib import Path
    from pryon_runtime_analysis import lfbe, metrics
    tap = 'StatefulPartitionedCall/functional_5/fc_output_ww/Tensordot/Reshape:0'
    model = onnx.load(native)
    model.graph.output.append(onnx.helper.make_tensor_value_info(tap, onnx.TensorProto.FLOAT, None))
    with tempfile.NamedTemporaryFile(suffix='.onnx') as f:
        onnx.save(model, f.name); sess = session(f.name)
    capture = Path(capture); stock = []
    for i, r in enumerate(json.loads((capture/'hits.json').read_text())):
        g = r.get('gemm')
        if g and (g['k'], g['n']) == (100, 3):
            stock.append(np.fromfile(capture/f'{i:05d}-x.bin', '<f4').reshape(g['m'], g['input_stride'])[:, :100])
    stock = np.array(stock)
    ours = np.array([o[tap] for o in stream(sess, lfbe(np.fromfile(audio, '<i2')), 0.0)])[:len(stock)].reshape(stock.shape)
    ok = np.isfinite(stock).all(-1) & np.isfinite(ours).all(-1); ok[:warmup] = False
    return {'cycles': len(stock), 'warmup_cycles_excluded': warmup,
            'classifier_input': metrics(ours[ok], stock[ok]),
            'stock_activation_rms': float(np.sqrt((stock[ok]**2).mean()))}


if __name__ == '__main__':
    import argparse, json
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('capture'); p.add_argument('audio')
    p.add_argument('--native', default='local-data/pryon-training/pryon-native-float.onnx')
    a = p.parse_args(); print(json.dumps(compare(a.native, a.audio, a.capture), indent=2))
