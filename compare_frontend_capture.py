#!/usr/bin/env python3
"""Read-only comparison at the normalized conv1 boundary. No captures/weights are exported."""
import argparse,json
from pathlib import Path
import numpy as np
import onnx
from onnx import numpy_helper
from pryon_runtime_analysis import lfbe


def recovered_normalized_frames(capture):
    frames=[]
    for i,r in enumerate(json.loads((capture/'hits.json').read_text())):
        g=r.get('gemm',{});p=capture/f'{i:05d}-x.bin'
        if (g.get('m'),g.get('k'),g.get('n'))!=(360,35,96) or not p.exists():continue
        w=np.fromfile(p,'<f4').reshape(360,g['input_stride'])[:,:35].reshape(6,60,7,5)
        if not np.array_equal(w[:,:-1,:,1:],w[:,1:,:,:-1],equal_nan=True):raise ValueError('Conv1 frequency overlap mismatch')
        frames.append(np.concatenate([w[:,0,6,:4],w[:,:,6,4]],axis=1))
    if not frames:raise ValueError('No captured conv1 inputs')
    return np.concatenate(frames)


def compare(model,capture,pcm):
    m=onnx.load(model);names=[m.graph.node[i].input[1] for i in (4,5)]
    init={t.name:numpy_helper.to_array(t) for t in m.graph.initializer if t.name in names}
    mean,inv=[init[n].ravel() for n in names]
    if mean.shape!=(64,) or inv.shape!=(64,):raise ValueError('Unexpected normalization tensors')
    ref=recovered_normalized_frames(capture)
    x=lfbe(np.fromfile(pcm,'<i2'))[:len(ref)];y=((x-mean)*inv).astype(np.float32)
    if y.shape!=ref.shape or not np.isfinite(ref).all():raise ValueError('Capture/audio length or nonfinite boundary mismatch')
    return dict(scope='normalized conv1 input; caller-supplied private capture, no stock execution',frames=len(ref),bins=64,
                exact_frames=int(np.all(y.view(np.uint32)==ref.view(np.uint32),axis=1).sum()),max_abs_error=float(np.max(abs(y-ref))))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',type=Path,required=True);p.add_argument('--capture',type=Path,required=True);p.add_argument('--pcm',type=Path,required=True)
    a=p.parse_args();print(json.dumps(compare(a.model,a.capture,a.pcm),indent=2))
