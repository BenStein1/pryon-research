"""Synthetic tests for safe original Pryon wake-head row quantization."""
from __future__ import annotations

import unittest
import numpy as np
from onnx import helper, TensorProto, numpy_helper

from pryon_head_quantize import (
    quantize_row, locate_head_and_bias, replace_original_pryon_wake_row
)


def fixture():
    q = np.arange(300, dtype=np.int16).reshape(3, 100)
    q = np.clip(q - 150, -128, 127).astype(np.int8)
    names = ("real_w", "real_w_s", "real_w_zp", "real_w_sums")
    init = [
        numpy_helper.from_array(q, name=names[0]),
        numpy_helper.from_array(
            np.array([.01, .02, .03], np.float32), name=names[1]),
        numpy_helper.from_array(
            np.array([0, 1, 2], np.int32), name=names[2]),
        numpy_helper.from_array(
            q.astype(np.int32).sum(axis=1), name=names[3]),
        numpy_helper.from_array(
            np.array([0.25, -0.1, 0.75], np.float32), name="head_bias"),
    ]
    nodes = [
        helper.make_node(
            "HybridGemm", ["activation"] + list(names), ["projected"],
            name="StatefulPartitionedCall/functional_5/fc_output_ww/Tensordot/MatMul"),
        helper.make_node(
            "Reshape", ["projected", "shape"], ["reshaped"], name="reshape"),
        helper.make_node(
            "Add", ["reshaped", "head_bias"], ["biased"], name="add"),
        helper.make_node(
            "Softmax", ["biased"], ["softmax_output_ww"],
            name="softmax"),
    ]
    model = helper.make_model(helper.make_graph(
        nodes, "test",
        [helper.make_tensor_value_info("activation", TensorProto.FLOAT, [100,100])],
        [helper.make_tensor_value_info("softmax_output_ww", TensorProto.FLOAT,
                                       [25, 4, 3])],
        initializer=init,
    ))
    return model


class QuantizedPryonHeadTests(unittest.TestCase):
    def test_signed_asymmetric_int8_roundtrip(self):
        w = np.linspace(-.75, 1.25, 100, dtype=np.float32)
        q, scale, zp, raw_sum, error = quantize_row(w)
        self.assertEqual(q.shape, (100,))
        self.assertEqual(q.dtype, np.int8)
        self.assertTrue(scale > 0)
        self.assertEqual(int(raw_sum), int(q.astype(np.int32).sum()))
        self.assertLessEqual(error, float(scale) * 0.75)

    def test_find_original_bias_and_wake_row(self):
        original = fixture()
        head, bias = locate_head_and_bias(original)
        self.assertIn("fc_output_ww", head.name)
        self.assertEqual(bias, "head_bias")

    def test_candidate_keeps_all_other_rows_and_source_unchanged(self):
        original = fixture()
        old_bytes = original.SerializeToString()
        new_w = np.linspace(-0.6, 0.8, 100, dtype=np.float32)
        candidate, info = replace_original_pryon_wake_row(
            original, new_w, 0.42, class_index=1
        )
        self.assertEqual(old_bytes, original.SerializeToString())
        self.assertTrue(info["original_architecture_preserved"])
        self.assertTrue(info["other_classes_preserved"])
        originals = {t.name: numpy_helper.to_array(t)
                     for t in original.graph.initializer}
        new = {t.name: numpy_helper.to_array(t)
               for t in candidate.graph.initializer}
        for name in ("real_w", "real_w_s", "real_w_zp", "real_w_sums", "head_bias"):
            self.assertTrue(np.array_equal(originals[name][0], new[name][0]))
            self.assertTrue(np.array_equal(originals[name][2], new[name][2]))
        self.assertNotEqual(float(originals["head_bias"][1]),
                            float(new["head_bias"][1]))
        q = new["real_w"][1]
        self.assertEqual(
            int(new["real_w_sums"][1]), int(q.astype(np.int32).sum()))

    def test_invalid_target_fails_without_touching_original(self):
        model = fixture()
        original = model.SerializeToString()
        with self.assertRaisesRegex(ValueError, "three classes"):
            replace_original_pryon_wake_row(
                model, np.zeros(100), 0, class_index=3)
        with self.assertRaisesRegex(ValueError, "finite"):
            replace_original_pryon_wake_row(
                model, np.ones(100), float("inf"), class_index=1)
        self.assertEqual(original, model.SerializeToString())


if __name__ == "__main__":
    unittest.main()
