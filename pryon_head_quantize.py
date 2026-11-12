#!/usr/bin/env python3
"""Build a Pryon-architecture-preserving quantized wake-head candidate.

This module edits ONE output class row in a **deep copy** of the genuine
Pryon ONNX: int8 weights, per-row float scale, int32 zero-point, int32
raw weight sum, and its separate Add bias. Other classes, operator nodes,
and all streaming state tensors are untouched.

Not a standalone detector, wrapper or surrogate network. Actual training
and libpryon acceptance testing are separate experiments. No Echo flashing.
"""
from __future__ import annotations

import copy
import numpy as np
import onnx
from onnx import numpy_helper

NAME = "fc_output_ww/Tensordot/MatMul"


def quantize_row(weights):
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != (100,) or not np.isfinite(w).all():
        raise ValueError("Expected finite 100-D original Pryon class weights")
    low = min(0.0, float(w.min()))
    high = max(0.0, float(w.max()))
    if high <= low:
        scale = 1e-8
    else:
        scale = max((high - low) / 255, 1e-8)
    zero_point = int(np.clip(np.rint(-128.0 - low / scale), -128, 127))
    q = np.clip(np.rint(w / scale + zero_point), -128, 127).astype(np.int8)
    reconstructed = (q.astype(np.float64) - zero_point) * scale
    return q, np.float32(scale), np.int32(zero_point), np.int32(
        q.astype(np.int32).sum()), float(np.max(np.abs(w - reconstructed)))


def locate_head_and_bias(model):
    inits = {x.name: x for x in model.graph.initializer}
    heads = [
        n for n in model.graph.node if n.op_type == "HybridGemm"
        and NAME in n.name
    ]
    if len(heads) != 1 or len(heads[0].input) != 5:
        raise RuntimeError("Expected one original 5-input Pryon output HybridGemm")
    head = heads[0]
    producers = {
        out: n for n in model.graph.node for out in n.output if out
    }
    consumers = {}
    for node in model.graph.node:
        for name in node.input:
            if name:
                consumers.setdefault(name, []).append(node)
    pending = [head.output[0]]
    visited = set()
    bias_name = None
    for _ in range(8):
        next_pending = []
        for tensor in pending:
            for node in consumers.get(tensor, []):
                if node.name in visited:
                    continue
                visited.add(node.name)
                if node.op_type == "Add":
                    options = [
                        x for x in node.input if x in inits
                        and list(inits[x].dims) == [3]
                        and inits[x].data_type == onnx.TensorProto.FLOAT
                    ]
                    if len(options) != 1:
                        raise RuntimeError(
                            f"Expected a single 3-element float bias, got {options}"
                        )
                    bias_name = options[0]
                    break
                next_pending.extend([name for name in node.output if name])
            if bias_name is not None:
                break
        if bias_name is not None or not next_pending:
            break
        pending = next_pending
    if bias_name is None:
        raise RuntimeError("No original 3-element Add bias after wake head")
    for name in head.input[1:]:
        if name not in inits:
            raise RuntimeError(f"Missing original head initializer: {name}")
    return head, bias_name


def replace_original_pryon_wake_row(model, weight, bias, class_index=1):
    if class_index not in (0, 1, 2):
        raise ValueError("Pryon original head has exactly three classes")
    if not np.isfinite(float(bias)):
        raise ValueError("Head bias must be finite")
    head, bias_name = locate_head_and_bias(model)
    tensors = {x.name: numpy_helper.to_array(x).copy()
               for x in model.graph.initializer
               if x.name in set(head.input[1:]) | {bias_name}}
    weight_name, scale_name, zp_name, sums_name = head.input[1:5]
    q = tensors[weight_name]
    scales, zps, sums = (
        tensors[scale_name], tensors[zp_name], tensors[sums_name]
    )
    biases = tensors[bias_name]
    if q.shape != (3, 100) or q.dtype != np.int8:
        raise RuntimeError("Unexpected original Pryon int8 classifier dimensions")
    if any(a.shape != (3,) for a in (scales, zps, sums, biases)):
        raise RuntimeError("Unexpected original Pryon row quantization shape")
    if not np.issubdtype(scales.dtype, np.floating):
        raise RuntimeError("Invalid original scale tensor dtype")
    if biases.dtype != np.float32:
        raise RuntimeError("Unexpected original bias storage type")

    replacement, scale, zero_point, row_sum, error = quantize_row(weight)
    new_tensors = {
        weight_name: q.copy(),
        scale_name: scales.copy(),
        zp_name: zps.copy(),
        sums_name: sums.copy(),
        bias_name: biases.copy(),
    }
    new_tensors[weight_name][class_index] = replacement
    new_tensors[scale_name][class_index] = scale
    new_tensors[zp_name][class_index] = zero_point
    new_tensors[sums_name][class_index] = row_sum
    new_tensors[bias_name][class_index] = np.float32(bias)

    candidate = copy.deepcopy(model)
    changed = []
    for tensor in candidate.graph.initializer:
        replacement_tensor = new_tensors.get(tensor.name)
        if replacement_tensor is not None:
            if not np.array_equal(tensors[tensor.name], replacement_tensor):
                changed.append(tensor.name)
            tensor.CopyFrom(numpy_helper.from_array(
                replacement_tensor, name=tensor.name
            ))
    if not changed:
        raise RuntimeError("Requested candidate does not change any weights or bias")
    source_names = [n.name for n in model.graph.node]
    if [n.name for n in candidate.graph.node] != source_names:
        raise RuntimeError("Original Pryon graph architecture changed unexpectedly")
    for name, before in tensors.items():
        after = new_tensors[name]
        if before.shape != after.shape or before.dtype != after.dtype:
            raise RuntimeError(f"Original quantization contract changed for {name}")
        if not np.array_equal(np.delete(before, class_index, axis=0),
                              np.delete(after, class_index, axis=0)):
            raise RuntimeError(f"Unexpected change to other Pryon classes in {name}")
    return candidate, {
        "class_index": class_index,
        "original_head_shape": [3, 100],
        "weight_quantization_max_abs_error": error,
        "new_class_scale": float(scale),
        "new_class_zero_point": int(zero_point),
        "new_class_int8_sum": int(row_sum),
        "changed_initializer_count": len(changed),
        "original_architecture_preserved": True,
        "other_classes_preserved": True,
        "real_Pryon_acceptance_verified": False,
    }
