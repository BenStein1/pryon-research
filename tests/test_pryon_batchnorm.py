"""Stock BN affine coefficients and single-rounding epilogue; synthetic only."""
import unittest
import numpy as np
from pryon_quant_model import Model


class BatchNormTests(unittest.TestCase):
    def test_affine_epilogue_rounds_once(self):
        model=Model.__new__(Model)
        model.init={name:np.array([value],np.float32) for name,value in
                    [('g',1-2**-23),('b',-1),('mu',0),('var',1)]}
        x=np.array([[[1+2**-23]]],np.float32)
        y=model.bn(x,['unused','g','b','mu','var'],0.)
        self.assertEqual(y.dtype,np.float32)
        self.assertEqual(float(y[0,0,0]),-2**-46)
        # Separate float32 multiplication/addition would lose this residue.
        self.assertEqual(float((x*model.init['g'][0]-1)[0,0,0]),0.)
