"""Native temporal loss/gradient parity; synthetic features, no vendor assets."""
import ctypes as C
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import numpy as np
from pryon_qat import prepare_rows, quantized_logits, window_loss
from pryon_head_quantize import quantize_row
from pryon_quant_model import quant_matmul
from train_pryon_streaming import temporal_windows, temporal_loss


class Head(C.Structure):
    _fields_ = [('q', (C.c_int8*100)*3), ('scale', C.c_float*3),
                ('zero', C.c_int32*3), ('bias', C.c_float*3)]


class Clip(C.Structure):
    _fields_ = [('cycles', C.c_int), ('label', C.c_int),
                ('x', C.POINTER(C.c_float)), ('scale', C.POINTER(C.c_float)),
                ('q', C.POINTER(C.c_int32)), ('constant', C.POINTER(C.c_ubyte)),
                ('base_float', C.POINTER(C.c_double)), ('base_quant', C.POINTER(C.c_double))]


class Fit(C.Structure):
    _fields_ = [('mean', C.c_double*100), ('std', C.c_double*100),
                ('prior', C.c_double*101), ('head', C.POINTER(Head))]


@unittest.skipUnless(shutil.which('cc'), 'C compiler required')
class NativeQATTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        source = Path(__file__).resolve().parents[1]/'native/pryon_qat.c'
        lib = Path(cls.tmp.name)/'qat.so'
        subprocess.run(['cc', '-O2', '-shared', '-fPIC', '-ffp-contract=off',
                        str(source), '-lm', '-o', str(lib)], check=True, capture_output=True)
        cls.lib = C.CDLL(str(lib))
        cls.lib.pq_prepare.argtypes = [C.POINTER(Clip), C.POINTER(Head)]
        cls.lib.pq_release.argtypes = [C.POINTER(Clip)]
        cls.lib.pq_evaluate.argtypes = [C.POINTER(Fit), C.POINTER(Clip), C.c_int,
            C.POINTER(C.c_int), C.c_int, C.POINTER(C.c_double), C.c_int, C.POINTER(C.c_double)]
        cls.lib.pq_evaluate.restype = C.c_double

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_native_float_and_quantized_objectives_keep_temporal_windows(self):
        rng = np.random.default_rng(8)
        arrays = [rng.normal(0, .4, (34, 25, 100)).astype(np.float32) for _ in range(2)]
        for a in arrays:
            a[:30] = np.nan  # Native initialization rows must never enter fitting.
            a[30, 0] = 0
            a[30, 1] = .2
        head = Head(); qw=[]; sw=[]; zw=[]
        for j in range(3):
            q,s,z,_,_ = quantize_row(rng.normal(0,.02,100).astype(np.float32))
            C.memmove(head.q[j], q.ctypes.data, 100)
            head.scale[j],head.zero[j],head.bias[j] = float(s),int(z),.1*j
            qw.append(q);sw.append(s);zw.append(z)
        qw,sw,zw = np.array(qw),np.array(sw),np.array(zw)
        clips = (Clip*2)()
        try:
            for i,a in enumerate(arrays):
                clips[i].cycles=34; clips[i].label=1-i
                clips[i].x=a.ctypes.data_as(C.POINTER(C.c_float))
                self.assertEqual(self.lib.pq_prepare(C.byref(clips[i]),C.byref(head)),0)
            theta=rng.normal(0,.02,101).astype(np.float64)
            fit=Fit();fit.head=C.pointer(head)
            for k in range(100):fit.std[k]=1.
            for k in range(101):fit.prior[k]=theta[k]
            native_grad=np.zeros(101,np.float64)
            for quantized in (0,1):
                got=self.lib.pq_evaluate(C.byref(fit),clips,2,None,2,
                    theta.ctypes.data_as(C.POINTER(C.c_double)),quantized,
                    native_grad.ctypes.data_as(C.POINTER(C.c_double)))
                windows=[temporal_windows(a).astype(np.float64) for a in arrays]
                weights=(qw.astype(float)-zw[:,None])*sw[:,None]
                if not quantized:
                    bases=[np.logaddexp(x@weights[0]+head.bias[0],x@weights[2]+head.bias[2]) for x in windows]
                    expected,grad=temporal_loss(theta,windows,bases,[1,0],theta.copy())
                else:
                    expected=0.;grad=np.zeros(101)
                    for i,x in enumerate(windows):
                        rows=prepare_rows(x)
                        base=quant_matmul(rows['x'],qw[[0,2]],sw[[0,2]],zw[[0,2]],{'constant_rows':0})+np.array([head.bias[0],head.bias[2]],np.float32)
                        base=np.logaddexp(base[:,0].astype(float),base[:,1].astype(float)).reshape(rows['shape'])
                        value,dp=window_loss(quantized_logits(rows,theta[:-1],theta[-1]),base,1-i,pool='smooth' if i==0 else 'max')
                        qdq=rows['q']*rows['s'][:,None];qdq[rows['const']]=rows['x'][rows['const']]
                        expected+=value/2;grad[:-1]+=dp.ravel()@qdq/2;grad[-1]+=dp.sum()/2
                self.assertAlmostEqual(got,expected,places=7)
                np.testing.assert_allclose(native_grad,grad,atol=2e-7,rtol=2e-6)
        finally:
            for clip in clips:
                clip.x=C.POINTER(C.c_float)() # NumPy owns this buffer; free only native caches.
                self.lib.pq_release(C.byref(clip))

    def test_exp_log_log1p_are_bit_identical_to_the_echos_libm(self):
        """Golden hash of the real Echo's bionic libm (Android 7.1.2) on 240,006 fixed inputs.

        Captured with local-data/libm-probe/probe.c on the device and, identically,
        from the firmware's libm.so under QEMU. glibc differs on ~7% of exp results.
        """
        import hashlib
        for name in ('pq_exp', 'pq_log', 'pq_log1p'):
            fn = getattr(self.lib, name); fn.restype = C.c_double; fn.argtypes = [C.c_double]
        r = np.random.default_rng(7)
        x = np.concatenate([r.uniform(-60, 0, 60000), r.uniform(-700, 700, 40000), r.uniform(0, 1, 40000),
                            np.exp(r.uniform(-40, 40, 40000)), r.uniform(-1+1e-12, 0, 40000),
                            10.0**r.uniform(-17, -1, 20000), [0.0, 1.0, -1.0, 0.5, 2.0, 1e-300]]).astype('<f8')
        self.assertEqual(len(x), 240006)
        out = np.array([[self.lib.pq_exp(v), self.lib.pq_log(v), self.lib.pq_log1p(v)] for v in x.tolist()], '<f8')
        bits = out.view('<u8'); bits[np.isnan(out)] = 0x7ff8000000000000  # NaN sign/payload is platform noise
        self.assertEqual(hashlib.sha256(bits.tobytes()).hexdigest(),
                         'b6ceac3aca5e78f579030a328eae4d5472003cd1ad4709501730cef7231d5e85')
