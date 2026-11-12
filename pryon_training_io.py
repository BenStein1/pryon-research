"""Portable model helpers extracted from the original research; no corpus or machine defaults."""
import copy, hashlib
from pathlib import Path
import numpy as np
from onnx import numpy_helper
from pryon_head_quantize import locate_head_and_bias

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def original_class_weights(model):
    node, bias_name = locate_head_and_bias(model)
    names = node.input[1:5]
    tensors = {t.name: numpy_helper.to_array(t)
               for t in model.graph.initializer
               if t.name in set(names) | {bias_name}}
    q, scale, zp = (
        tensors[names[0]].astype(np.float64),
        tensors[names[1]].astype(np.float64),
        tensors[names[2]].astype(np.float64),
    )
    w = (q - zp[:, None]) * scale[:, None]
    b = tensors[bias_name].astype(np.float64)
    if w.shape != (3, 100) or b.shape != (3,):
        raise RuntimeError("Source Pryon head is no longer [3,100] + [3] bias")
    return w, b

def check_original_head_only(source_model, candidate_model):
    """Prove all graph bytes outside the five class-row tensors are identical."""
    source_head, source_bias = locate_head_and_bias(source_model)
    candidate_head, candidate_bias = locate_head_and_bias(candidate_model)
    allowed = set(source_head.input[1:5]) | {source_bias}
    if (source_head.name != candidate_head.name
            or source_bias != candidate_bias
            or list(source_head.input) != list(candidate_head.input)):
        raise RuntimeError("Candidate changed the original classifier topology")
    source_inits = list(source_model.graph.initializer)
    candidate_inits = list(candidate_model.graph.initializer)
    if [t.name for t in source_inits] != [t.name for t in candidate_inits]:
        raise RuntimeError("Candidate changed original initializer inventory")
    reconstructed = copy.deepcopy(candidate_model)
    changed = 0
    for index, (old, new) in enumerate(zip(source_inits, candidate_inits)):
        if old.name in allowed:
            a, b = numpy_helper.to_array(old), numpy_helper.to_array(new)
            if (a.shape != b.shape or a.dtype != b.dtype
                    or a.shape[0] != 3
                    or not np.array_equal(
                        np.delete(a, 1, axis=0),
                        np.delete(b, 1, axis=0))):
                raise RuntimeError("Candidate changed a non-target classifier row")
            if not np.array_equal(a[1], b[1]):
                changed += 1
            reconstructed.graph.initializer[index].CopyFrom(old)
    if reconstructed.SerializeToString() != source_model.SerializeToString():
        raise RuntimeError("Candidate modifies non-classifier model structure or data")
    if changed == 0:
        raise RuntimeError("Candidate has no trained classifier-row changes")
    return changed
