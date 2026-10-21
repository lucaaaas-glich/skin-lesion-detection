import unittest
from preflight import normalize_device, validate_probe_size


class PreflightTests(unittest.TestCase):
    def test_missing_cuda_does_not_silently_change_experiment_device(self):
        with self.assertRaises(RuntimeError): normalize_device('0', False, 0)
        with self.assertRaises(RuntimeError): normalize_device('cuda:1', True, 1)
        self.assertEqual(normalize_device('0', True, 1), 'cuda:0')
        self.assertEqual(normalize_device('cpu', False, 0), 'cpu')

    def test_cpu_probe_is_explicitly_small_and_gpu_shape_is_valid(self):
        with self.assertRaises(ValueError): validate_probe_size('cpu', 16, 640)
        with self.assertRaises(ValueError): validate_probe_size('cuda:0', 16, 641)
        validate_probe_size('cpu', 2, 128)
        validate_probe_size('cuda:0', 16, 640)


if __name__ == '__main__':
    unittest.main()
