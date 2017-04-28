import h5py
import numpy as np
import nutcracker
import unittest



class TestCaseIntensity(unittest.TestCase):
    def test_fourier_shell_correlation_2d(self):
        img = np.random.random((10,10))
        
        fsc_calculated = nutcracker.intensities.fourier_shell_correlation(img,img)
        fsc_expected = np.ones_like(fsc_calculated.shape)
        
        self.assertTrue(np.alltrue(np.round(fsc_calculated-fsc_expected,7) == 0))
        

