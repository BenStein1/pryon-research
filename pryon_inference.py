"""File/PCM inference adapter; no enrollment UI or optimizer dependencies.

The measured Model.stream engine remains the shared six-frame/25-lane core.
Padding here is the desktop enrollment clip protocol, not a continuous-stream
reset policy. Embedded continuous audio should drive Model.stream directly.
"""
import numpy as np
from pryon_decoder import decode
from pryon_runtime_analysis import lfbe


def streaming_activations(model, pcm):
    # Deterministic fresh stream: 2s silence primes the 25-lane ring; 2s tail
    # lets the measured trim/lock decoder finish. No reset between cycles.
    padded = np.pad(pcm, (32000, 32000))
    a = np.array([o['classifier_input'].copy() for o in model.stream(lfbe(padded))])
    if a.ndim != 3 or a.shape[1:] != (25, 100) or not np.isfinite(a).all():
        raise RuntimeError('Invalid streaming classifier activations')
    return a


def replay_head(model, a):
    logits = model.dense('fc_output_ww', a.reshape(-1, 100), {'constant_rows': 0})
    e = np.exp(logits-logits.max(1, keepdims=True))
    p = (e/e.sum(1, keepdims=True)).reshape(len(a), 25, 3)
    return list(decode(p))
