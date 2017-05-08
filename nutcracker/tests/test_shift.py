import h5py
import numpy as np
import nutcracker
import condor
import unittest
import os

_data_dir = os.path.dirname(os.path.realpath(__file__)) + "/../data"
with h5py.File(_data_dir + '/test_data.h5', 'r') as f:
    img_1 = f['real'][:]
with h5py.File(_data_dir + '/test_data_rot_shift.h5', 'r') as f:
    img_2 = f['real'][:]

class TestCaseShift(unittest.TestCase):
    def test_find_shift_between_two_models(self):
        out_calculated = nutcracker.utils.shift.find_shift_between_two_models(img_2,img_1,rotation_angles=[0.52359878,0.52359878,0.52359878])
        out_expected = np.array((2,-3,1))

        self.assertTrue(np.alltrue(np.round(out_calculated-out_expected) == 0))
