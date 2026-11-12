#!/usr/bin/env python3
"""Experimental frozen-head training using the measured Pryon streaming engine.

Manifest: [{"path":"clip.wav", "label":1, "split":"train"}, ...].
Paths are relative to the manifest. Both train and test must contain positives
and negatives. Output is restricted to ignored local-data; source stays intact.
No empirical LFBE offset, native float translation, QEMU or device operations.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from scipy.io import wavfile
from scipy.optimize import minimize
from scipy.special import expit, logsumexp

from pryon_paths import DEFAULT_SOURCE
from pryon_training_io import original_class_weights, sha256
from pryon_inference import streaming_activations, replay_head
from pryon_head_quantize import replace_original_pryon_wake_row
from pryon_quant_model import Model

ROOT = Path(__file__).resolve().parent


def read_manifest(path):
    rows = json.loads(path.read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError('Manifest must be a nonempty list')
    seen = set()
    for row in rows:
        if row['label'] not in (0, 1) or row['split'] not in ('train', 'test', 'diagnostic'):
            raise ValueError('Expected binary label and train/test/diagnostic split')
        audio = (path.parent / row['path']).resolve()
        digest = sha256(audio)
        if digest in seen:
            raise ValueError('Duplicate audio content: possible train/test leakage')
        seen.add(digest)
        row.update(path=str(audio), sha256=digest)
    for split, minimum in (('train', 5), ('test', 1)):
        for label in (0, 1):
            if sum(r['split'] == split and r['label'] == label for r in rows) < minimum:
                raise ValueError(f'Need >= {minimum} {split} recordings per label')
    return rows


def temporal_windows(a):
    """[cycles, 15 eligible windows, 3 ages, 100 features], after warmup.

    The cycle and lane axes are retained; the three probabilities in each
    decoder window must be averaged before selecting a detection score.
    """
    return np.array([[a[t, [(t-age-j+1) % 25 for j in range(3)]]
                      for age in range(9, 24)] for t in range(30, len(a))])


def temporal_loss(theta, windows, baselines, labels, prior, temperature=.02, reg=1e-5):
    """Smooth-max of decoder window probabilities, with recording-level BCE.

    Softmax class-1 probabilities are computed against the two fixed classes.
    A normalized logsumexp approximates the decoder max during optimization;
    inference still uses the exact max, lock and threshold in pryon_decoder.
    """
    loss, grad = 0., np.zeros_like(theta)
    for x, baseline, label in zip(windows, baselines, labels):
        p = expit(x @ theta[:-1] + theta[-1] - baseline)
        scores = p.mean(axis=-1)
        z = scores / temperature
        lse = logsumexp(z)
        score = temperature * (lse - np.log(scores.size))
        score = np.clip(score, 1e-12, 1-1e-12)
        loss -= (label*np.log(score)+(1-label)*np.log1p(-score))/len(labels)
        dscore = (score-label)/(score*(1-score)*len(labels))
        dp = dscore * np.exp(z-lse)[..., None] * p*(1-p)/3
        grad[:-1] += np.sum(dp[..., None]*x, axis=(0,1,2))
        grad[-1] += dp.sum()
    delta = theta[:-1]-prior[:-1]
    return loss + .5*reg*(delta @ delta), grad + np.r_[reg*delta, 0.]


def optimize_temporal(prior, xs, baselines, labels, regularization=1e-5):
    """Continue a valid iteration-limited fit; never silently accept nonconvergence."""
    from functools import partial
    if not np.isfinite(regularization) or regularization <= 0:
        raise ValueError('Regularization must be finite and positive')
    objective = partial(temporal_loss, reg=regularization)
    theta = prior
    attempts = []
    for budget in (350, 1400):
        result = minimize(objective, theta, args=(xs,baselines,labels,prior),
                          jac=True, method='L-BFGS-B',
                          options={'maxiter':budget,'ftol':1e-10,'maxls':35})
        attempts.append(dict(iterations=int(result.nit), message=str(result.message),
                             loss=float(result.fun), success=bool(result.success)))
        if not np.isfinite(result.x).all() or not np.isfinite(result.fun):
            break
        if result.success:
            return result, attempts
        # scipy status 1 can also mean maxfun. Retry only an iteration limit;
        # abnormal line searches and other failures need separate diagnosis.
        if result.status != 1 or 'ITERATIONS REACHED LIMIT' not in str(result.message):
            break
        theta = result.x
    raise RuntimeError(f'Temporal optimizer failed after {sum(a["iterations"] for a in attempts)} iterations: {result.message}')


def train_temporal(activations, labels, weights, biases, regularization=1e-5):
    windows = [temporal_windows(a).astype(np.float64) for a in activations]
    # Normalization improves conditioning but is folded into exported weights.
    flat = np.concatenate([x.reshape(-1,100) for x in windows])
    mean, std = flat.mean(0), np.maximum(flat.std(0), 1e-3)
    baselines = [np.logaddexp(x @ weights[0]+biases[0],
                             x @ weights[2]+biases[2]) for x in windows]
    xs = [(x-mean)/std for x in windows]
    prior = np.r_[weights[1]*std, biases[1]+mean @ weights[1]]
    result, attempts = optimize_temporal(prior, xs, baselines, labels, regularization)
    w = result.x[:-1]/std
    b = result.x[-1]-mean @ w
    return w.astype(np.float32), np.float32(b), dict(
        objective='recording BCE of smooth-max decoder window probabilities',
        smooth_max_temperature=.02, regularization=regularization,
        optimizer_iterations=sum(a["iterations"] for a in attempts),
        optimizer_attempts=attempts, optimizer_final_loss=float(result.fun),
        optimizer_success=bool(result.success))


def train_experiment(manifest, source, output, qat=False):
    """Train/export from a portable recording manifest; no microphone/UI calls."""
    source = Path(source)
    out = Path(output).resolve()
    if not out.is_relative_to(ROOT/'local-data'):
        raise ValueError('Output must be inside ignored project local-data')
    if out.exists():
        raise ValueError('Use a new output directory to preserve prior experiments')
    rows = read_manifest(Path(manifest).resolve())
    source_hash = sha256(source)
    model = Model(source)
    original = onnx.load(source)
    weights, biases = original_class_weights(original)
    out.mkdir(parents=True)
    activations = [None] * len(rows)
    def extract(i):
        row = rows[i]
        rate, pcm = wavfile.read(row['path'])
        if rate != 16000 or pcm.dtype != np.int16 or pcm.ndim != 1:
            raise ValueError('Require mono 16-kHz signed-int16 WAV')
        a = streaming_activations(model, pcm)
        activations[i] = a
        np.save(out/f'activations-{i}.npy', a)
        print(f'extracted {i+1}/{len(rows)}: {len(a)} cycles', flush=True)
    train = [i for i, row in enumerate(rows) if row['split'] == 'train']
    for i in train:
        extract(i)
    w, b, fit = train_temporal([activations[i] for i in train],
                             [rows[i]['label'] for i in train], weights, biases)
    if qat:
        from pryon_qat import train_qat
        head = model.gemm['fc_output_ww']
        fixed = tuple(model.init[head[j]][[0, 2]] for j in (1, 2, 3))
        print('Float TRAIN-only initialization complete; fitting 1600-step temporal int8 QAT.', flush=True)
        w, b, quantized_fit = train_qat([activations[i] for i in train],
            [rows[i]['label'] for i in train], fixed, biases[[0, 2]].astype(np.float32),
            w, b, steps=1600, negative_pool='max', batch_size=8,
            class_balance=True, eval_interval=50, seed=103)
        fit = dict(quantized_fit, float_train_initialization=fit)
    candidate, export = replace_original_pryon_wake_row(original, w, b)
    candidate_path = out/'candidate.onnx'
    onnx.save(candidate, candidate_path)
    frozen_hash = sha256(candidate_path)
    print('Candidate frozen; extracting reserved evaluation recordings.', flush=True)
    for i, a in enumerate(activations):
        if a is None:
            extract(i)
    fitted = Model(candidate_path)
    for row, a in zip(rows, activations):
        events = replay_head(fitted, a)
        row['events'] = [list(e) for e in events]
        row['accepted'] = any(e[0] == 2 for e in events)
    counts = {}
    for split in ('train', 'test', 'diagnostic'):
        group = [r for r in rows if r['split'] == split]
        counts[split] = {name: sum(r['label'] == label and r['accepted'] == accept for r in group)
                         for name, label, accept in [('tp',1,True),('fn',1,False),('fp',0,True),('tn',0,False)]}
    if sha256(source) != source_hash or sha256(candidate_path) != frozen_hash:
        raise RuntimeError('Source or frozen candidate changed during evaluation')
    report = dict(source_sha256=source_hash, candidate_sha256=sha256(candidate_path),
                  encoder_sha256=sha256(Path(__file__).with_name('pryon_quant_model.py')),
                  frontend='measured LFBE; no offset', lifecycle='6 frames; 25-lane ring; 2s prefix/tail',
                  fit=fit, export=export, counts=counts, recordings=rows,
                  actual_libpryon_verified=False,
                  limitations=['Frozen Alexa encoder may not represent arbitrary phrases',
                               'Temporal loss does not optimize decoder lock timing; positive pooling uses smooth max',
                               'Heldout clips restart independently; not continuous-room false-accept measurement',
                               'Speaker independence depends on caller-provided split'])
    (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(counts, indent=2))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest', type=Path)
    p.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--qat', action='store_true', help='Int8-aware temporal fitting after TRAIN-only float initialization')
    args = p.parse_args()
    train_experiment(args.manifest, args.source, args.out, qat=args.qat)


if __name__ == '__main__':
    main()
