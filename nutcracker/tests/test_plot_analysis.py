import numpy as np
import nutcracker
import unittest

class TestCasePlotAnalysis(unittest.TestCase):
    def test_envelope(self):
        x = np.arange(100)
        y = np.sinc(x * 0.05)

        upper, lower = nutcracker.utils.plot_analysis.envelope(y,1,order_spline_interpolation=3,peak_finding_threshold=(0,1))

        self.assertTrue(np.alltrue(np.round(y - upper,7) == 0) and np.alltrue(np.round(y - lower,7) == 0))
        
