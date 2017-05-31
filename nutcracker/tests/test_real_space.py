import spimage
import numpy as np
import nutcracker
import unittest

class TestCaseRealSpace(unittest.TestCase):
    def test_phase_retieval_transfer_function(self):
        img = np.random.random((10,10,10))
        sup = np.ones((10,10,10))

        img = np.array([img,img])
        sup = np.array([sup,sup])

        prtf_calculated = nutcracker.real_space.phase_retieval_transfer_function(img,sup,full_output=True)
        prtf_calculated = prtf_calculated['prtf_3D_volume']
        prtf_expected = np.ones((10,10,10))

        self.assertTrue(np.alltrue(np.round(prtf_calculated-prtf_expected,6) == 0))
