#!/usr/bin/env python3
"""Best-effort native ONNX translation of proprietary Pryon Hybrid* operators.

Creates a separate, local-only FLOAT32 graph for feature extraction and
training. The original ONNX and Amazon firmware are never modified.

Warning: Pryon's exact quantized-kernel and LSTM semantics are not publicly
specified; dequantization formula and ONNX gate order are hypotheses. Native
graph loading does NOT prove equivalence with real Pryon inference.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
from pryon_paths import DEFAULT_SOURCE
WORK = HERE / "local-data/pryon-training"
HYBRID = {"HybridConv", "HybridLSTM", "HybridGemm"}


def read_attr(onnx, node, name, default=None):
    for attr in node.attribute:
        if attr.name == name:
            return onnx.helper.get_attribute_value(attr)
    return default


def dequant(onnx, inits, weight, scale, zero_point):
    from onnx import numpy_helper

    def arr(name):
        if name not in inits:
            raise RuntimeError(f"Missing quantized tensor initializer: {name}")
        try:
            return numpy_helper.to_array(inits[name])
        except Exception as exc:
            raise RuntimeError(f"Cannot decode initializer {name}: {exc}") from exc

    q = np.asarray(arr(weight))
    s = np.asarray(arr(scale), dtype=np.float32)
    zp = np.asarray(arr(zero_point), dtype=np.float32)
    if q.dtype != np.int8:
        raise RuntimeError(f"Expected INT8 weight {weight}, got {q.dtype}")
    if q.ndim == 2 and s.shape == (q.shape[0],):
        s = s[:, None]
        zp = zp[:, None]
    elif q.ndim == 3 and s.shape == q.shape[:2]:
        s = s[..., None]
        zp = zp[..., None]
    elif s.size == 1:
        pass
    else:
        raise RuntimeError(f"Unrecognized per-output quantization shapes: "
                           f"{q.shape} / {s.shape} / {zp.shape} for {weight}")
    out = (q.astype(np.float32) - zp) * s
    if not np.isfinite(out).all():
        raise RuntimeError(f"Dequantized tensor contains nonfinite values: {weight}")
    return out.astype(np.float32)


def spatial_attr(onnx, node, name, default):
    vals = read_attr(onnx, node, name, default)
    vals = list(vals)
    if len(vals) == 4:
        vals = vals[2:]
    if len(vals) != 2:
        raise RuntimeError(f"{node.name}: {name} expects 2 or 4 values, got {vals}")
    return [int(v) for v in vals]


def broadcast_lstm_input(onnx, new_nodes, new_inits, x, initial_h, prefix):
    """Experiment: expand X batch=1 to the batch carried by initial_h.

    In the original Pryon model both LSTM branches receive X=[T,1,100]
    and state=[1,25,100]. Standard ONNX LSTM requires X=[T,B,100]
    with state=[1,B,100]. The proprietary HybridLSTM may broadcast
    input X over B parallel state lanes. Build this expansion dynamically
    from the X and state tensor shapes, rather than hardcoding B=25.

    This is only a float native test hypothesis, NOT proven Pryon parity.
    """
    from onnx import numpy_helper

    idx = []
    for i in range(3):
        name = f"{prefix}_idx_{i}"
        new_inits.append(numpy_helper.from_array(
            np.array([i], dtype=np.int64), name=name))
        idx.append(name)
    x_shape = f"{prefix}_x_shape"
    h_shape = f"{prefix}_h_shape"
    seq = f"{prefix}_seq"
    batch = f"{prefix}_batch"
    features = f"{prefix}_features"
    expanded_shape = f"{prefix}_target_shape"
    expanded_x = f"{prefix}_expanded_x"
    new_nodes.extend([
        onnx.helper.make_node(
            "Shape", [x], [x_shape], name=f"{prefix}_Shape_X"
        ),
        onnx.helper.make_node(
            "Shape", [initial_h], [h_shape],
            name=f"{prefix}_Shape_initial_h"
        ),
        onnx.helper.make_node(
            "Gather", [x_shape, idx[0]], [seq],
            name=f"{prefix}_Gather_X_time", axis=0
        ),
        onnx.helper.make_node(
            "Gather", [h_shape, idx[1]], [batch],
            name=f"{prefix}_Gather_state_batch", axis=0
        ),
        onnx.helper.make_node(
            "Gather", [x_shape, idx[2]], [features],
            name=f"{prefix}_Gather_X_features", axis=0
        ),
        onnx.helper.make_node(
            "Concat", [seq, batch, features], [expanded_shape],
            name=f"{prefix}_Concat_target_shape", axis=0
        ),
        onnx.helper.make_node(
            "Expand", [x, expanded_shape], [expanded_x],
            name=f"{prefix}_Expand_X_batch"
        ),
    ])
    return expanded_x


def translate(onnx, original, output, *, broadcast_lstm_batch=False):
    """Convert 7 Conv + 2 LSTM + 4 Gemm kernels to native ONNX operators."""
    from onnx import numpy_helper

    model = copy.deepcopy(original)
    inits = {i.name: i for i in model.graph.initializer}
    new_inits = []
    new_nodes = []
    report = []
    seen = set()
    for index, n in enumerate(model.graph.node):
        if n.op_type not in HYBRID:
            new_nodes.append(copy.deepcopy(n))
            continue
        seen.add(n.op_type)
        name = n.name or f"{n.op_type}_{index}"
        x = list(n.input)
        prefix = f"__pryon_native_{index}"
        if n.op_type == "HybridGemm":
            if len(x) != 5 or not all(x[:5]):
                raise RuntimeError(f"{name}: expected X, W, scale, zp, sums")
            w = dequant(onnx, inits, x[1], x[2], x[3])
            if w.ndim != 2:
                raise RuntimeError(f"{name}: expected 2D Gemm weights, got {w.shape}")
            key = prefix + "_w"
            new_inits.append(numpy_helper.from_array(w, name=key))
            new_nodes.append(onnx.helper.make_node(
                "Gemm", [x[0], key], list(n.output),
                name=name + "__native", transB=1))
            report.append({"type": "HybridGemm", "node": name, "weight_shape": list(w.shape),
                           "replacement": "Gemm(transB=1)"})

        elif n.op_type == "HybridConv":
            if len(x) != 6 or not all(x[:6]):
                raise RuntimeError(f"{name}: expected X, W, scale, zp, sums, bias")
            q = dequant(onnx, inits, x[1], x[2], x[3])
            if q.ndim != 2:
                raise RuntimeError(f"{name}: expected flattened 2D Conv weights")
            fmt = read_attr(onnx, n, "data_format", b"NCHW")
            if isinstance(fmt, bytes):
                fmt = fmt.decode("ascii", "replace")
            if fmt != "NCHW":
                raise RuntimeError(f"{name}: unsupported data_format={fmt!r}; "
                                   "need explicit data transpositions")
            kernel = spatial_attr(onnx, n, "kernel_shape", None)
            group = int(read_attr(onnx, n, "group", 1))
            channels_per_group = int(q.shape[1] // np.prod(kernel))
            if channels_per_group * np.prod(kernel) != q.shape[1]:
                raise RuntimeError(f"{name}: incompatible flattened Conv weight "
                                   f"{list(q.shape)} with kernel={kernel}")
            w = q.reshape((q.shape[0], channels_per_group, *kernel))
            bias = numpy_helper.to_array(inits[x[5]]).astype(np.float32)
            if bias.size != q.shape[0]:
                raise RuntimeError(f"{name}: bias shape {bias.shape} mismatches {q.shape}")
            keyw, keyb = prefix + "_w", prefix + "_b"
            new_inits.extend([numpy_helper.from_array(w, name=keyw),
                              numpy_helper.from_array(bias.reshape(-1), name=keyb)])
            attrs = {
                "kernel_shape": kernel,
                "group": group,
                "strides": spatial_attr(onnx, n, "strides", (1, 1)),
                "dilations": spatial_attr(onnx, n, "dilations", (1, 1)),
            }
            new_nodes.append(onnx.helper.make_node(
                "Conv", [x[0], keyw, keyb], list(n.output),
                name=name + "__native", **attrs))
            report.append({"type": "HybridConv", "node": name,
                           "weight_shape": list(w.shape), "attributes": attrs,
                           "replacement": "Conv"})

        elif n.op_type == "HybridLSTM":
            # Pryon HybridLSTM signature:
            # X, Wq, Ws, Wzp, Wsums, Rq, Rs, Rzp, Rsums,
            # B, sequence_lens?, initial_h, initial_c
            if len(x) != 13:
                raise RuntimeError(f"{name}: expected 13 HybridLSTM inputs, got {len(x)}")
            W = dequant(onnx, inits, x[1], x[2], x[3])
            R = dequant(onnx, inits, x[5], x[6], x[7])
            if W.ndim != 3 or R.ndim != 3 or W.shape[:2] != R.shape[:2]:
                raise RuntimeError(f"{name}: incompatible LSTM weights {W.shape} {R.shape}")
            bias = numpy_helper.to_array(inits[x[9]]).astype(np.float32)
            if list(bias.shape) != [1, W.shape[1] * 2]:
                raise RuntimeError(f"{name}: LSTM bias shape {bias.shape} does not match "
                                   f"expected [1, {W.shape[1]*2}]")
            kw, kr, kb = prefix + "_W", prefix + "_R", prefix + "_B"
            new_inits.extend([numpy_helper.from_array(W, name=kw),
                              numpy_helper.from_array(R, name=kr),
                              numpy_helper.from_array(bias, name=kb)])
            attrs = {"hidden_size": int(read_attr(onnx, n, "hidden_size", 100))}
            direction = read_attr(onnx, n, "direction", b"forward")
            if isinstance(direction, bytes):
                direction = direction.decode("ascii", "replace")
            attrs["direction"] = direction
            activations = read_attr(onnx, n, "activations", None)
            if activations:
                attrs["activations"] = [
                    a.decode("ascii", "replace") if isinstance(a, bytes) else a
                    for a in activations
                ]
            if not x[11]:
                raise RuntimeError(
                    f"{name}: expected explicit initial_h to size the state batch"
                )

            # Measured stock Pryon behavior:
            # One input vector drives all 25 LSTM state rows.
            # The streaming driver handles the per-cycle state reset.
            if broadcast_lstm_batch:
                # Retain the existing optional converter implementation.
                lstm_x = broadcast_lstm_input(
                    onnx, new_nodes, new_inits, x[0], x[11], prefix
                )
            else:
                # REA's measured default implementation.
                state_shape = prefix + "_state_shape"
                lstm_x = prefix + "_x_batch"

                new_nodes.append(onnx.helper.make_node(
                    "Shape",
                    [x[11]],
                    [state_shape],
                    name=name + "__state_shape",
                ))
                new_nodes.append(onnx.helper.make_node(
                    "Expand",
                    [x[0], state_shape],
                    [lstm_x],
                    name=name + "__x_batch",
                ))

            new_nodes.append(onnx.helper.make_node(
                "LSTM",
                [lstm_x, kw, kr, kb, x[10], x[11], x[12]],
                list(n.output), name=name + "__native", **attrs))
            report.append({"type": "HybridLSTM", "node": name,
                           "broadcast_input_batch": True,
                           "W": list(W.shape), "R": list(R.shape),
                           "bias": list(bias.shape), "attributes": attrs,
                           "replacement": "Expand(X to state batch) + LSTM"})
    actual = {e["type"] for e in report}
    if not HYBRID.issubset(actual):
        raise RuntimeError(f"Missing expected Hybrid operators: {HYBRID - actual}")
    if len(report) != 13:
        raise RuntimeError(f"Expected 13 Hybrid nodes, converted {len(report)}; "
                           "the extracted firmware may differ.")

    # The source exports some Reshape shape inputs as INT32, which Pryon's
    # private runtime accepts but ONNX Runtime does not. Convert their shape
    # inputs with explicit native-only casts without altering original values.
    types = {i.name: i.data_type for i in model.graph.initializer}
    for value in list(model.graph.input) + list(model.graph.value_info):
        if value.type.HasField("tensor_type"):
            types[value.name] = value.type.tensor_type.elem_type
    rewrite = []
    casts = []
    for index, node in enumerate(new_nodes):
        if node.op_type == "Reshape" and len(node.input) > 1 and types.get(
                node.input[1]) == onnx.TensorProto.INT32:
            clone = copy.deepcopy(node)
            old = clone.input[1]
            dst = f"__pryon_native_shape_{index}"
            rewrite.append(onnx.helper.make_node(
                "Cast", [old], [dst], name=f"native_shape_cast_{index}",
                to=onnx.TensorProto.INT64))
            clone.input[1] = dst
            rewrite.append(clone)
            casts.append({"node": node.name, "old": old})
        else:
            rewrite.append(node)

    model.graph.ClearField("node")
    model.graph.node.extend(rewrite)
    model.graph.initializer.extend(new_inits)
    normalization = normalize_legacy_native_graph(onnx, model)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output), save_as_external_data=False)
    return {
        "original_nodes": len(original.graph.node),
        "translated_hybrid_nodes": len(report),
        "experimental_lstm_batch_broadcast": bool(broadcast_lstm_batch),
        "native_reshape_casts": len(casts),
        "legacy_normalization": normalization,
        "translations": report,
        "casts": casts,
        "destination": str(output),
        "warning": "FLOAT32 approximation; not yet proven numerically equivalent to Pryon.",
    }


def normalize_legacy_native_graph(onnx, model):
    """Repair Pryon-specific ONNX schema quirks in the native-only copy.

    1. Some streaming state placeholders appear as BOTH graph inputs and
       initializer descriptors, but their TensorProto has no value bytes.
       Remove only those empty initializer entries, retaining graph inputs
       so callers still supply the state explicitly.
    2. Pryon's Shape extension has a dtype attribute. ONNX Shape always
       returns int64. Replace Shape(dtype=int32) with standard Shape ->
       Cast(int32), preserving the original consumer-visible tensor name.
    """
    inputs = {v.name for v in model.graph.input}
    retained = []
    missing = []
    for init in model.graph.initializer:
        elems = 1
        for dim in init.dims:
            elems *= dim
        has_data = bool(
            init.raw_data or init.float_data or init.int32_data
            or init.int64_data or init.double_data or init.uint64_data
            or init.string_data or init.external_data
        )
        # An initializer with a zero-sized dimension legitimately has no
        # values. Do not remove it. Only remove positive-sized placeholders.
        if elems > 0 and not has_data:
            if init.name not in inputs:
                raise RuntimeError(
                    f"Initializer {init.name!r} has no serialized data and "
                    "is not a graph input. Cannot safely convert this model."
                )
            missing.append(init.name)
            continue
        retained.append(copy.deepcopy(init))
    model.graph.ClearField("initializer")
    model.graph.initializer.extend(retained)

    known_names = {v.name for v in model.graph.input}
    known_names.update(v.name for v in model.graph.output)
    known_names.update(v.name for v in model.graph.value_info)
    known_names.update(v.name for v in model.graph.initializer)
    known_names.update(name for node in model.graph.node for name in node.output)

    rewrite = []
    fixed_shapes = []
    for index, original_node in enumerate(model.graph.node):
        if original_node.op_type != "Shape":
            rewrite.append(copy.deepcopy(original_node))
            continue
        dtype_attrs = [attr for attr in original_node.attribute
                       if attr.name == "dtype"]
        if not dtype_attrs:
            rewrite.append(copy.deepcopy(original_node))
            continue
        if len(dtype_attrs) != 1:
            raise RuntimeError(f"Shape node {original_node.name!r} has "
                               "multiple dtype attributes")
        dtype = int(onnx.helper.get_attribute_value(dtype_attrs[0]))
        if dtype not in (onnx.TensorProto.INT32, onnx.TensorProto.INT64):
            raise RuntimeError(
                f"Unsupported Pryon Shape(dtype={dtype}) on node "
                f"{original_node.name!r}; expected int32 or int64."
            )
        shape = copy.deepcopy(original_node)
        shape.ClearField("attribute")
        shape.attribute.extend(
            copy.deepcopy(attr) for attr in original_node.attribute
            if attr.name != "dtype"
        )
        if dtype == onnx.TensorProto.INT32:
            if len(shape.output) != 1:
                raise RuntimeError(
                    f"Cannot preserve Shape output count for {shape.name!r}"
                )
            output_name = shape.output[0]
            temporary = f"__pryon_native_shape_int64_{index}"
            while temporary in known_names:
                temporary += "_"
            known_names.add(temporary)
            shape.output[0] = temporary
            rewrite.append(shape)
            rewrite.append(onnx.helper.make_node(
                "Cast", [temporary], [output_name],
                name=f"native_shape_dtype_cast_{index}",
                to=onnx.TensorProto.INT32,
            ))
        else:
            rewrite.append(shape)
        fixed_shapes.append({
            "node": original_node.name,
            "requested_dtype": "int32" if dtype == onnx.TensorProto.INT32
                               else "int64",
        })
    model.graph.ClearField("node")
    model.graph.node.extend(rewrite)

    # Standard ONNX shape/index operands need int64 in several operators.
    # Pryon's old exporter emits int32 values for some of these (including
    # Tile repeat counts). Insert casts only on the native-only graph copy,
    # retaining the original tensor names and values for their other users.
    type_by_name = {v.name: v.data_type for v in model.graph.initializer}
    for value in (list(model.graph.input) + list(model.graph.value_info)
                  + list(model.graph.output)):
        if value.type.HasField("tensor_type"):
            type_by_name[value.name] = value.type.tensor_type.elem_type
    for node in model.graph.node:
        if node.op_type == "Cast" and node.output:
            target = next((int(onnx.helper.get_attribute_value(a))
                           for a in node.attribute if a.name == "to"), None)
            if target is not None:
                type_by_name[node.output[0]] = target

    # Positions are only appropriate for opsets with input-tensor axes
    # (our extracted Pryon model uses that schema).
    int64_operands = {
        "Tile": (1,),
        "Unsqueeze": (1,),
        "Squeeze": (1,),
        "Expand": (1,),
        "ConstantOfShape": (0,),
        "TopK": (1,),
    }
    rewritten = []
    fixed_index_inputs = []
    for index, old_node in enumerate(model.graph.node):
        node = copy.deepcopy(old_node)
        for input_index in int64_operands.get(node.op_type, ()):
            if input_index >= len(node.input) or not node.input[input_index]:
                continue
            source = node.input[input_index]
            if type_by_name.get(source) != onnx.TensorProto.INT32:
                continue
            temporary = f"__pryon_native_int64_input_{index}_{input_index}"
            while temporary in known_names:
                temporary += "_"
            known_names.add(temporary)
            rewritten.append(onnx.helper.make_node(
                "Cast", [source], [temporary],
                name=f"native_int64_input_cast_{index}_{input_index}",
                to=onnx.TensorProto.INT64,
            ))
            node.input[input_index] = temporary
            type_by_name[temporary] = onnx.TensorProto.INT64
            fixed_index_inputs.append({
                "node": old_node.name or f"{old_node.op_type}#{index}",
                "operator": old_node.op_type,
                "input_index": input_index,
                "source": source,
            })
        rewritten.append(node)
    model.graph.ClearField("node")
    model.graph.node.extend(rewritten)

    return {
        "empty_initializer_placeholders_removed": len(missing),
        "placeholder_names": missing,
        "custom_shape_dtype_nodes_rewritten": len(fixed_shapes),
        "shape_dtype_nodes": fixed_shapes,
        "int64_index_shape_inputs_cast": len(fixed_index_inputs),
        "int64_index_shape_cast_details": fixed_index_inputs,
    }


def validate_ort(onnx, output):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    result = {"onnx_checker": "not-run", "onnxruntime": "not-run"}
    try:
        onnx.checker.check_model(str(output))
        result["onnx_checker"] = "passed"
    except Exception as exc:
        result["onnx_checker"] = f"{type(exc).__name__}: {str(exc)[:750]}"
    try:
        sess = ort.InferenceSession(
            str(output), sess_options=opts, providers=["CPUExecutionProvider"])
        result["onnxruntime"] = "loaded"
        result["required_inputs"] = [
            {"name": v.name, "shape": v.shape, "type": v.type}
            for v in sess.get_inputs()
        ]
        result["outputs"] = [o.name for o in sess.get_outputs()]
    except Exception as exc:
        result["onnxruntime"] = f"{type(exc).__name__}: {str(exc)[:1600]}"
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--out", type=Path, default=WORK / "pryon-native-float.onnx")
    ap.add_argument("--lstm-broadcast", action="store_true",
                    help="EXPERIMENT: broadcast LSTM input batch=1 to the "
                         "parallel state batch size (25 in this model). "
                         "Native-only, not verified Pryon parity.")
    args = ap.parse_args()
    import onnx
    if not args.model.is_file():
        raise RuntimeError(f"Original extracted model missing: {args.model}")
    model = onnx.load(str(args.model), load_external_data=True)
    conversion = translate(onnx, model, args.out,
                           broadcast_lstm_batch=args.lstm_broadcast)
    print(f"Converted {conversion['translated_hybrid_nodes']} Pryon custom ops and "
          f"inserted {conversion['native_reshape_casts']} native-only shape casts.")
    norm = conversion["legacy_normalization"]
    print(f"Normalized native copy: removed "
          f"{norm['empty_initializer_placeholders_removed']} empty initializer "
          f"descriptors (preserved as graph inputs); rewrote "
          f"{norm['custom_shape_dtype_nodes_rewritten']} nonstandard Shape dtype "
          f"attributes; cast {norm['int64_index_shape_inputs_cast']} int32 "
          f"index/shape inputs to int64.")
    print(f"Experimental LSTM input-batch broadcast: "
          f"{bool(args.lstm_broadcast)}")
    print(f"Native-only graph: {args.out}")
    outcome = validate_ort(onnx, args.out)
    report = {"conversion": conversion, "validation": outcome}
    dest = args.out.parent / "native-conversion-report.json"
    dest.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(f"ONNX checker: {outcome['onnx_checker']}")
    print(f"ONNX Runtime: {outcome['onnxruntime']}")
    print(f"Report: {dest}")
    if outcome["onnxruntime"] != "loaded":
        raise RuntimeError(
            "Native conversion is not executable yet; inspect the error above. "
            "The original model was not modified, and no training has occurred.")
    print("Native copy loads, but numerical equivalence to Pryon is UNVERIFIED. "
          "Next: validate the encoder on reference audio and train in that feature space.")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, OSError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
