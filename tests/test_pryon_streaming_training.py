"""Training uses the same aged ring rows and quantized head as inference."""
import unittest
import numpy as np
from train_pryon_streaming import temporal_windows, temporal_loss, replay_head


class StreamingTrainingTests(unittest.TestCase):
    def test_qat_freezes_before_extracting_test_recordings(self):
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch
        from train_pryon_streaming import ROOT, train_experiment
        rows = [dict(path=f'{split}-{label}', split=split, label=label)
                for split in ('train', 'test') for label in (1, 0)]
        stages = []
        weights, bias = np.zeros((3, 100), np.float32), np.zeros(3, np.float32)
        model = SimpleNamespace(gemm={'fc_output_ww': ['x', 'w', 's', 'z']},
                                init=dict(w=np.zeros((3, 100), np.int8),
                                          s=np.ones(3, np.float32), z=np.zeros(3, np.uint8)))
        def read(path):
            stages.append(path)
            if path.startswith('test'):
                self.assertIn('frozen', stages)
            return 16000, np.ones(16000, np.int16)
        def fit(aa, labels, *args, **kwargs):
            self.assertEqual(labels, [1, 0])
            self.assertEqual(len(aa), 2)
            self.assertFalse(any(s.startswith('test') for s in stages))
            stages.append('fit')
            return weights[1], bias[1], {}
        def save(candidate, path):
            Path(path).write_bytes(b'synthetic candidate')
            stages.append('frozen')
        with tempfile.TemporaryDirectory(dir=ROOT/'local-data') as d:
            source = Path(d)/'source'; source.write_bytes(b'synthetic source')
            with patch.multiple('train_pryon_streaming',
                    read_manifest=lambda _: [dict(r) for r in rows], Model=lambda _: model,
                    original_class_weights=lambda _: (weights, bias),
                    streaming_activations=lambda *a: np.zeros((31, 25, 100), np.float32),
                    train_temporal=fit, replace_original_pryon_wake_row=lambda *a: (None, {}),
                    replay_head=lambda *a: []), \
                 patch('train_pryon_streaming.wavfile.read', side_effect=read), \
                 patch('train_pryon_streaming.onnx.load', return_value=None), \
                 patch('train_pryon_streaming.onnx.save', side_effect=save), \
                 patch('pryon_qat.train_qat', side_effect=fit) as qat:
                train_experiment(Path(d)/'manifest', source, Path(d)/'output', qat=True)
            self.assertEqual(qat.call_count, 1)
            self.assertEqual(stages[:5], ['train-1', 'train-0', 'fit', 'fit', 'frozen'])

    def test_aged_rows_keep_absolute_cycle(self):
        a = np.arange(32*25*100).reshape(32,25,100)
        rows = temporal_windows(a)
        self.assertEqual(rows.shape, (2,15,3,100))
        np.testing.assert_array_equal(rows[0,0,0], a[30,22])  # age 9
        np.testing.assert_array_equal(rows[0,14,2], a[30,6])  # age 25
        np.testing.assert_array_equal(rows[1,0,0], a[31,23])

    def test_temporal_gradient(self):
        from scipy.optimize import approx_fprime
        rng = np.random.default_rng(3)
        xs = [rng.normal(size=(2,15,3,100)) for _ in range(2)]
        baselines = [np.zeros((2,15,3)) for _ in xs]
        theta = rng.normal(0,.01,101)
        args = (xs, baselines, [1,0], theta.copy())
        _, gradient = temporal_loss(theta,*args)
        numerical = approx_fprime(theta,lambda t: temporal_loss(t,*args)[0],1e-7)
        np.testing.assert_allclose(gradient,numerical,atol=1e-6)

    def test_iteration_limit_continues_from_last_weights(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from train_pryon_streaming import optimize_temporal
        prior = np.zeros(3)
        last = np.ones(3)
        limited = SimpleNamespace(x=last, fun=1., success=False, nit=350,
            status=1, message='STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT')
        done = SimpleNamespace(x=last*2, fun=.5, success=True, nit=42,
            status=0, message='CONVERGENCE')
        with patch('train_pryon_streaming.minimize', side_effect=[limited,done]) as fit:
            result, attempts = optimize_temporal(prior, [], [], [], regularization=.001)
        self.assertIs(result,done)
        self.assertEqual(fit.call_args.args[0].keywords['reg'],.001)
        self.assertEqual(sum(a['iterations'] for a in attempts),392)
        np.testing.assert_array_equal(fit.call_args_list[1].args[1],last)
        np.testing.assert_array_equal(fit.call_args_list[1].kwargs['args'][-1],prior)

    def test_abnormal_optimizer_failure_is_not_accepted_or_retried(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from train_pryon_streaming import optimize_temporal
        failed = SimpleNamespace(x=np.zeros(3), fun=1., success=False, nit=2,
            status=2, message='ABNORMAL')
        with patch('train_pryon_streaming.minimize',return_value=failed) as fit:
            with self.assertRaisesRegex(RuntimeError,'ABNORMAL'):
                optimize_temporal(np.zeros(3),[],[],[])
        self.assertEqual(fit.call_count,1)

    def test_replay_uses_quantized_head_and_stock_decoder(self):
        class FakeModel:
            def dense(self, key, a, stats):
                self.key = key
                return np.column_stack([np.zeros(len(a)), a[:,0], np.zeros(len(a))])
        m = FakeModel()
        a = np.zeros((40,25,100), np.float32)
        a[30:,:,0] = 10
        events = replay_head(m,a)
        self.assertEqual(m.key,'fc_output_ww')
        self.assertEqual(events[0][0],2)
        self.assertEqual(events[0][2],960*31)
        self.assertEqual(events[0][3],30)
