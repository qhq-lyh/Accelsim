import tempfile
import unittest
from pathlib import Path

import numpy as np

from infer_frequency_matrix import load_triplets, threshold_frequency


class MatrixInverseTests(unittest.TestCase):
    def test_cubic_inverse(self):
        frequency, index = threshold_frequency(np.array([45., 45.]), np.array([10., 5.]), 80., 1000., 3.)
        self.assertAlmostEqual((frequency / 1000.) ** 3, 3.5)
        self.assertEqual(index, 0)

    def test_limiting_node_changes_with_fixed_background(self):
        base = np.array([79., 45.])
        rise = np.array([0.1, 20.])
        self.assertEqual(int(np.argmax(base + rise)), 0)
        frequency, index = threshold_frequency(base, rise, 80., 1000., 3.)
        self.assertEqual(index, 1)
        predicted = base + rise * (frequency / 1000.) ** 3
        self.assertAlmostEqual(predicted.max(), 80.)
        self.assertTrue(np.all(predicted <= 80. + 1e-10))

    def test_unreachable_background(self):
        with self.assertRaises(ValueError):
            threshold_frequency(np.array([81.]), np.array([1.]), 80., 1000., 3.)

    def test_dimensions_sentinel_is_not_a_nonzero(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'Bmatrix'
            path.write_text('0 0 1\n1 1 0.5\n3 2 0\n')
            matrix = load_triplets(path)
        self.assertEqual(matrix.shape, (3, 2))
        self.assertEqual(matrix.nnz, 2)
        np.testing.assert_allclose(matrix.toarray(), [[1., 0.], [0., 0.5], [0., 0.]])


if __name__ == '__main__':
    unittest.main()
