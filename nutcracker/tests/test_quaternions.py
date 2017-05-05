import h5py
import numpy as np
import nutcracker
import condor
import unittest
import os

_data_dir = os.path.dirname(os.path.realpath(__file__)) + "/../data"
with h5py.File(_data_dir + '/test_quaternions.h5', 'r') as f:
    q1 = f['quaternion'][:]
    q2 = f['quaternion_rot'][:]
    q_relative = f['quaternion_relative'][:]

class TestCaseQuaternions(unittest.TestCase):
    def test_compare_two_sets_of_quaternions(self):
        p, out = nutcracker.quaternions.compare_two_sets_of_quaternions(q1,q2,n_samples=10, full_output=True, sigma=3)
    
        self.assertGreaterEqual(p,0.99)


    def test_global_quaternion_rotation_between_two_sets(self):
        out = nutcracker.quaternions.global_quaternion_rotation_between_two_sets(q1,q2,full_output=True)
        quat_rel = out['quat_array_mean']

        self.assertTrue(np.alltrue(np.round(quat_rel - q_relative,7) == 0))
