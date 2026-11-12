"""Synthetic unit tests for experimental Pryon LSTM batch expansion.

No Amazon model, device or recordings are accessed.
"""
from __future__ import annotations

import unittest

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

from convert_pryon_native import broadcast_lstm_input


def session_for_batch(batch):
    new_nodes, new_inits = [], []
    expanded_x = broadcast_lstm_input(
        onnx, new_nodes, new_inits, "x", "h", "fixture"
    )
    new_nodes.append(helper.make_node(
        "LSTM",
        [expanded_x, "W", "R", "B", "", "h", "c"],
        ["Y", "Y_h", "Y_c"],
        name="native_lstm", hidden_size=100,
    ))
    new_inits.extend([
        numpy_helper.from_array(
            np.zeros((1, 400, 100), dtype=np.float32), name="W"),
        numpy_helper.from_array(
            np.zeros((1, 400, 100), dtype=np.float32), name="R"),
        numpy_helper.from_array(
            np.zeros((1, 800), dtype=np.float32), name="B"),
    ])
    graph = helper.make_graph(
        new_nodes, "native_broadcast_fixture",
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [4, 1, 100]),
            helper.make_tensor_value_info("h", TensorProto.FLOAT,
                                          [1, batch, 100]),
            helper.make_tensor_value_info("c", TensorProto.FLOAT,
                                          [1, batch, 100]),
        ],
        [
            helper.make_tensor_value_info(
                expanded_x, TensorProto.FLOAT, [4, batch, 100]),
            helper.make_tensor_value_info(
                "Y", TensorProto.FLOAT, [4, 1, batch, 100]),
            helper.make_tensor_value_info(
                "Y_h", TensorProto.FLOAT, [1, batch, 100]),
            helper.make_tensor_value_info(
                "Y_c", TensorProto.FLOAT, [1, batch, 100]),
        ],
        initializer=new_inits,
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", 17)]
    )
    model.ir_version = 9
    onnx.checker.check_model(model)
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    sess = ort.InferenceSession(
        model.SerializeToString(), sess_options=opts,
        providers=["CPUExecutionProvider"]
    )
    return sess, expanded_x


class LSTMBroadcastTests(unittest.TestCase):
    def test_expands_single_stream_without_losing_frames(self):
        session, expanded_name = session_for_batch(25)
        x = np.arange(4 * 100, dtype=np.float32).reshape(4, 1, 100) * 0.01
        feed = {
            "x": x, "h": np.zeros((1, 25, 100), np.float32),
            "c": np.zeros((1, 25, 100), np.float32),
        }
        expanded, output, h, c = session.run(
            [expanded_name, "Y", "Y_h", "Y_c"], feed)
        self.assertEqual(expanded.shape, (4, 25, 100))
        self.assertEqual(output.shape, (4, 1, 25, 100))
        self.assertEqual(h.shape, (1, 25, 100))
        self.assertEqual(c.shape, (1, 25, 100))
        np.testing.assert_array_equal(expanded[:, 0, :], x[:, 0, :])
        np.testing.assert_array_equal(expanded[:, 24, :], x[:, 0, :])
        self.assertTrue(np.isfinite(output).all())

    def test_batch_is_inferred_dynamically_from_state(self):
        session, expanded_name = session_for_batch(7)
        x = np.ones((4, 1, 100), np.float32)
        expanded, output = session.run(
            [expanded_name, "Y"],
            {
                "x": x,
                "h": np.zeros((1, 7, 100), np.float32),
                "c": np.zeros((1, 7, 100), np.float32),
            },
        )
        self.assertEqual(expanded.shape, (4, 7, 100))
        self.assertEqual(output.shape, (4, 1, 7, 100))


if __name__ == "__main__":
    unittest.main()
