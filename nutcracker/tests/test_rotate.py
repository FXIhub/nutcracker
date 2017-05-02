import h5py
import numpy as np
import nutcracker
import unittest

with h5py.File('/home/doctor/software/nutcracker/nutcracker/data/test_data.h5', 'r') as f:
    img_1 = f['real'][:]
with h5py.File('/home/doctor/software/nutcracker/nutcracker/data/test_data_rot_shift.h5', 'r') as f:
    img_2 = f['real'][:]

class TestCaseRotate(unittest.TestCase):
    def test_find_rotation_between_two_models(self):
        Img_1 = np.abs(np.fft.fftshift(np.fft.fftn(img_1)))**2
        Img_2 = np.abs(np.fft.fftshift(np.fft.fftn(img_2)))**2
        
        out_calculated = nutcracker.utils.rotate.find_rotation_between_two_models(Img_2,Img_1,method='brute_force',
                                                                       number_of_evaluations=20,
                                                                       radius_radial_mask=20./2,
                                                                       order_spline_interpolation=3)
        out_expected = np.array((0.52359878,0.52359878,0.52359878))

        self.assertTrue(np.alltrue(np.round(out_calculated-out_expected,2) == 0))

    def test_find_shift_between_two_models(self):
        out_calculated = nutcracker.utils.rotate.find_shift_between_two_models(img_2,img_1)
        out_expected = np.array((2,-3,1))
        
        self.assertTrue(np.alltrue(np.round(out_calculated-out_expected) == 0))

    
#    def test_rotation_based_on_euler_angles(self):
        
