import json,tempfile,unittest
from pathlib import Path
import numpy as np
from compare_frontend_capture import recovered_normalized_frames
from demo import demo

class ReleaseTests(unittest.TestCase):
    def test_firmware_free_demo(self):
        r=demo();self.assertTrue(r['synthetic_only']);self.assertFalse(r['vendor_encoder_executed'])
        self.assertEqual(r['temporal_window_shape'],[3,15,3,100])
    def test_capture_boundary_recovery(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);x=np.arange(6*7*64,dtype=np.float32).reshape(6,7,64)
            w=np.stack([x[:,:,i:i+5] for i in range(60)],axis=1)
            w.reshape(360,35).astype('<f4').tofile(p/'00000-x.bin')
            (p/'hits.json').write_text(json.dumps([{'gemm':dict(m=360,k=35,n=96,input_stride=35)}]))
            np.testing.assert_array_equal(recovered_normalized_frames(p),x[:,-1])
