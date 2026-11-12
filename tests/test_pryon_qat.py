"""Synthetic QAT forward parity, smooth logit gradient, and actual learning."""
import unittest
import numpy as np
from scipy.optimize import approx_fprime
from pryon_head_quantize import quantize_row
from pryon_quant_model import quant_matmul
from pryon_qat import prepare_rows,quantized_logits,window_loss,train_qat


class QATTests(unittest.TestCase):
    def test_forward_matches_independent_kernel_including_constant_rows(self):
        rng=np.random.default_rng(4)
        x=rng.normal(size=(2,3,100)).astype(np.float32)
        x[0,0]=0;x[0,1]=.125;x[0,2]=-.25
        w=rng.normal(size=100).astype(np.float32)
        q,s,z,_,_=quantize_row(w)
        expected=quant_matmul(x.reshape(-1,100),q[None],np.array([s]),np.array([z]),{'constant_rows':0})[:,0]+np.float32(.3)
        np.testing.assert_array_equal(quantized_logits(prepare_rows(x),w,.3).ravel(),expected)

    def test_smooth_logit_gradient(self):
        x=np.linspace(-1,1,12).reshape(2,2,3)
        base=np.full_like(x,.3)
        for label in (0,1):
            for pool in ('smooth','max'):
                _,g=window_loss(x,base,label,pool=pool)
                numeric=approx_fprime(x.ravel(),lambda v:window_loss(v.reshape(x.shape),base,label,pool=pool)[0],1e-7)
                np.testing.assert_allclose(g.ravel(),numeric,atol=1e-6)

    def test_worst_window_is_not_diluted_by_long_negative_recording(self):
        x=np.full((80,15,3),-10.)
        x[40,7]=4.
        baseline=np.zeros_like(x)
        smooth,_=window_loss(x,baseline,0)
        hard,g=window_loss(x,baseline,0,pool='max')
        self.assertGreater(hard,smooth)
        self.assertEqual(np.count_nonzero(g),3)
        self.assertTrue(np.all(g[40,7]>0))

    def test_qat_improves_quantized_training_loss(self):
        rng=np.random.default_rng(9)
        aa=[rng.uniform(.1,1,(33,25,100)).astype(np.float32) for _ in range(2)]
        aa[0][...,0]=2;aa[1][...,0]=0
        competitors=(np.zeros((2,100),np.int8),np.ones(2,np.float32),np.zeros(2,np.int32))
        w,b,report=train_qat(aa,[1,0],competitors,np.zeros(2,np.float32),np.zeros(100,np.float32),-2.,steps=60,learning_rate=.03)
        self.assertEqual(w.dtype,np.float32)
        self.assertTrue(np.isfinite(w).all())
        self.assertLess(report['exported_loss'],report['initial_loss']-.05)


    def test_balanced_minibatches_checkpoint_full_class_balanced_loss(self):
        rng=np.random.default_rng(21)
        aa=[rng.uniform(.1,1,(33,25,100)).astype(np.float32) for _ in range(4)]
        for i,x in enumerate(aa):x[...,0]=2 if i<3 else 0
        fixed=(np.zeros((2,100),np.int8),np.ones(2,np.float32),np.zeros(2,np.int32))
        _,_,r=train_qat(aa,[1,1,1,0],fixed,np.zeros(2,np.float32),np.zeros(100,np.float32),-2,
                        steps=60,learning_rate=.03,batch_size=4,class_balance=True,eval_interval=10,seed=7)
        p=np.exp(-2)/(2+np.exp(-2))
        expected=(-np.log(p)-np.log1p(-p))/2
        self.assertAlmostEqual(r['initial_loss'],expected,places=10)
        self.assertLess(r['exported_loss'],r['initial_loss']-.05)
        self.assertTrue(r['class_balance'])
