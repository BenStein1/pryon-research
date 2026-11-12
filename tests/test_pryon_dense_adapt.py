"""Synthetic all-class gradient, constrained export, and learning checks."""
import unittest
import numpy as np
from onnx import numpy_helper
from scipy.optimize import approx_fprime
from tests.test_pryon_head_quantize import fixture
from pryon_dense_adapt import DENSE_BIAS,logits_loss_gradient,replace_dense_bias_and_head,check_dense_bias_and_head,train_dense


class DenseAdaptTests(unittest.TestCase):
    def test_all_class_logit_gradient(self):
        x=np.linspace(-1,1,36).reshape(2,2,3,3)
        for label in (0,1):
            _,g=logits_loss_gradient(x,label)
            numerical=approx_fprime(x.ravel(),lambda v:logits_loss_gradient(v.reshape(x.shape),label)[0],1e-7)
            np.testing.assert_allclose(g.ravel(),numerical,atol=1e-6)
        self.assertGreater(np.max(abs(g[...,0])),0)
        self.assertGreater(np.max(abs(g[...,2])),0)

    def test_export_guards_every_other_tensor_and_preserves_source(self):
        source=fixture()
        source.graph.initializer.append(numpy_helper.from_array(np.zeros(100,np.float32),name=DENSE_BIAS))
        source.graph.initializer.append(numpy_helper.from_array(np.ones(2,np.float32),name='frozen_encoder'))
        original=source.SerializeToString()
        candidate,_=replace_dense_bias_and_head(source,np.ones(100,np.float32),np.linspace(-1,1,100),.2)
        check_dense_bias_and_head(source,candidate)
        self.assertEqual(source.SerializeToString(),original)
        tensor=next(t for t in candidate.graph.initializer if t.name=='frozen_encoder')
        tensor.CopyFrom(numpy_helper.from_array(np.zeros(2,np.float32),name='frozen_encoder'))
        with self.assertRaises(RuntimeError):check_dense_bias_and_head(source,candidate)

    def test_training_reduces_exact_quantized_loss(self):
        rng=np.random.default_rng(13)
        base=[rng.normal(0,.1,(33,25,100)).astype(np.float32) for _ in range(2)]
        base[0][...,0]=1.;base[1][...,0]=-.5
        head=(np.zeros((3,100),np.int8),np.ones(3,np.float32),np.zeros(3,np.int32),np.zeros(3,np.float32))
        bh,w,b,_,info=train_dense(base,[1,0],head,np.zeros(100,np.float32),np.zeros(100,np.float32),-2,steps=60,learning_rate=.03)
        self.assertLess(info['best_loss'],info['initial_loss']-.01)
        self.assertTrue(np.isfinite(bh).all())
        self.assertEqual(bh.dtype,np.float32)

    def test_matrix_export_preserves_quantization_metadata_and_checks_sums(self):
        from onnx import helper
        from pryon_dense_adapt import replace_dense_layer_and_head,check_dense_layer_and_head
        source=fixture()
        names=['dense_q','dense_s','dense_z','dense_sum']
        arrays=[np.zeros((100,100),np.int8),np.ones(100,np.float32),np.zeros(100,np.int32),np.zeros(100,np.int32)]
        for name,a in zip(names,arrays):source.graph.initializer.append(numpy_helper.from_array(a,name=name))
        source.graph.initializer.append(numpy_helper.from_array(np.zeros(100,np.float32),name=DENSE_BIAS))
        source.graph.node.insert(0,helper.make_node('HybridGemm',['h']+names,['activation'],name='scope/fc_ww_1/MatMul'))
        original=source.SerializeToString()
        matrix=arrays[0].copy();matrix[0,0]=1
        candidate=replace_dense_layer_and_head(source,matrix,np.zeros(100),np.linspace(-1,1,100),.2)
        self.assertEqual(original,source.SerializeToString())
        check_dense_layer_and_head(source,candidate)
        tensor=next(t for t in candidate.graph.initializer if t.name=='dense_sum')
        tensor.CopyFrom(numpy_helper.from_array(np.zeros(100,np.int32),name='dense_sum'))
        with self.assertRaisesRegex(ValueError,'sums'):check_dense_layer_and_head(source,candidate)

    def test_matrix_updates_reduce_quantized_loss(self):
        rng=np.random.default_rng(17)
        h=[rng.normal(0,.1,(33,25,100)).astype(np.float32) for _ in range(2)]
        h[0][...,0]=2;h[1][...,0]=-2
        head=(np.zeros((3,100),np.int8),np.ones(3,np.float32),np.zeros(3,np.int32),np.zeros(3,np.float32))
        matrix=(np.zeros((100,100),np.int8),np.full(100,.02,np.float32),np.zeros(100,np.int32))
        initial=np.zeros(100,np.float32);initial[0]=.2
        _,_,_,q,info=train_dense([np.zeros_like(x) for x in h],[1,0],head,np.full(100,.2,np.float32),
            initial,-2,steps=80,learning_rate=.03,hidden_inputs=h,dense_matrix=matrix)
        self.assertLess(info['best_loss'],info['initial_loss']-.01)
        self.assertGreater(np.count_nonzero(q),0)
