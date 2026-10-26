from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[2] / 'NCC_SIAM_26'


class TimedBlockTests(unittest.TestCase):
    def test_restart_and_plots(self):
        with tempfile.TemporaryDirectory() as temporary:
            whole = Path(temporary) / 'whole'
            resumed = Path(temporary) / 'resumed'

            def sample(base, samples):
                subprocess.run([
                    sys.executable, str(SCRIPTS / 'timed_blocks.py'),
                    '--output', str(base), '--samples', str(samples),
                    '--runs', '1', '--taus', '0.05', '--Ts', '0.1',
                ], check=True, capture_output=True, text=True)

            sample(whole, 400)
            sample(resumed, 200)
            originals = {path: path.read_bytes() for path in resumed.rglob('block_*.npz')}
            sample(resumed, 400)
            for path, contents in originals.items():
                self.assertEqual(path.read_bytes(), contents)

            metadata = json.loads((resumed / 'metadata.json').read_text())
            precision = np.asarray(metadata['precision'])
            covariance = np.asarray(metadata['covariance'])
            np.testing.assert_allclose(precision, [[1, 511/513], [511/513, 1]])
            np.testing.assert_allclose(precision @ covariance, np.eye(2), atol=1e-13)
            self.assertAlmostEqual(np.linalg.cond(precision), 512)

            for method in ['LF', 'AA', 'N_K', 'FPI']:
                folder = Path(method) / 'tau_0.05' / 'T_0.1' / 'run_0'
                with np.load(resumed / folder / 'block_00000000.npz') as first:
                    previous = first['end_qp']
                with np.load(resumed / folder / 'block_00000001.npz') as second:
                    np.testing.assert_array_equal(previous, second['start_qp'])
                for index in range(2):
                    file = folder / f'block_{index:08d}.npz'
                    with np.load(whole / file) as expected, np.load(resumed / file) as actual:
                        for field in ['q', 'p', 'deltaHs', 'accepted', 'end_qp', 'next_key']:
                            np.testing.assert_array_equal(actual[field], expected[field])
                        self.assertEqual(actual['q'].shape, (200, 2))
                        self.assertTrue(np.isfinite(actual['q']).all())
                        self.assertGreater(float(actual['runtime']), 0)

            environment = dict(os.environ, MPLCONFIGDIR='/tmp/hmc_mpl_cov')
            subprocess.run([
                sys.executable, str(SCRIPTS / 'timed_blocks_plots.py'), '--output', str(resumed),
            ], check=True, capture_output=True, text=True, env=environment)
            self.assertEqual(len(list((resumed / 'plots').glob('*.png'))), 8)
            with np.load(resumed / 'cov_metrics.npz') as metrics:
                for method in metadata['methods']:
                    folder = resumed / method / 'tau_0.05' / 'T_0.1' / 'run_0'
                    blocks = []
                    elapsed = 0.0
                    for index in range(2):
                        with np.load(folder / f'block_{index:08d}.npz') as block:
                            blocks.append(block['q'])
                            elapsed += float(block['runtime'])
                        self.assertAlmostEqual(metrics[f'{method}_runtime'][0, 0, index, 0], elapsed)
                        error = np.cov(np.concatenate(blocks), rowvar=False) - covariance
                        self.assertAlmostEqual(metrics[f'{method}_cov_metric'][0, 0, index, 0],
                                               np.max(np.abs(np.diag(error))))
                        self.assertAlmostEqual(metrics[f'{method}_full_cov_metric'][0, 0, index, 0],
                                               np.max(np.abs(error)))


if __name__ == '__main__':
    unittest.main()
