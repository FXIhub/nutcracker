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

class TestCaseRotate(unittest.TestCase):
    def test_find_rotation_between_two_models_brute_force(self):
        Img_1 = np.abs(np.fft.fftshift(np.fft.fftn(img_1)))**2
        Img_2 = np.abs(np.fft.fftshift(np.fft.fftn(img_2)))**2
        
        out_calculated = nutcracker.utils.rotate.find_rotation_between_two_models(Img_2,Img_1,method='brute_force',
                                                                       number_of_evaluations=20,
                                                                       radius_radial_mask=20./2,
                                                                       order_spline_interpolation=3,
                                                                       full_output=True)
        out_calculated = out_calculated['rotation_angles']
        out_expected = np.array((0.52359878,0.52359878,0.52359878))

        self.assertTrue(np.alltrue(np.round(out_calculated-out_expected,2) == 0))

    def test_find_rotation_between_two_models_fmin_l_bfgs_b(self):
        Img_1 = np.abs(np.fft.fftshift(np.fft.fftn(img_1)))**2
        Img_2 = np.abs(np.fft.fftshift(np.fft.fftn(img_2)))**2

        out_calculated = nutcracker.utils.rotate.find_rotation_between_two_models(Img_2,Img_1,method='fmin_l_bfgs_b',
                                                                       number_of_evaluations=20,
                                                                       radius_radial_mask=20./2,
                                                                       order_spline_interpolation=3,
                                                                       initial_guess=[0.4,0.4,0.4],
                                                                       full_output=True)
        out_calculated = out_calculated['rotation_angles']
        out_expected = np.array((0.52359878,0.52359878,0.52359878))

        self.assertTrue(np.alltrue(np.round(out_calculated-out_expected,2) == 0))

    def test_find_rotation_between_two_models_fmin_l_bfgs_b(self):
        Img_1 = np.abs(np.fft.fftshift(np.fft.fftn(img_1)))**2
        Img_2 = np.abs(np.fft.fftshift(np.fft.fftn(img_2)))**2

        out_calculated = nutcracker.utils.rotate.find_rotation_between_two_models(Img_2,Img_1,method='fmin_l_bfgs_b',
                                                                       number_of_evaluations=20,
                                                                       radius_radial_mask=20./2,
                                                                       order_spline_interpolation=3,
                                                                       initial_guess=[0.4,0.4,0.4])
        out_expected = np.array((0.52359878,0.52359878,0.52359878))

        self.assertTrue(np.alltrue(np.round(out_calculated-out_expected,2) == 0))

    def test_rotation_based_on_rotation_matrix(self):
        angles = np.array((0.2,-4.2,1.8))
        
        r_z = nutcracker.utils.rotate.rotation_matrix(angles[0],'z')
        r_y = nutcracker.utils.rotate.rotation_matrix(angles[1],'y')
        r_x = nutcracker.utils.rotate.rotation_matrix(angles[2],'x')
        
        rot_mat = np.dot(np.dot(r_z,r_y),r_x)

        img_1_rot = nutcracker.utils.rotate.rotation_based_on_rotation_matrix(img_1,rot_mat)
        img_1_rot_back = nutcracker.utils.rotate.rotation_based_on_rotation_matrix(img_1_rot,np.transpose(rot_mat))
        
        self.assertTrue(np.mean(np.abs(img_1 - img_1_rot_back)**2) <=  1E-3)
        
    def test_rotation_based_on_quaternion(self):
        quaternion = condor.utils.rotation.rand_quat()

        img_1_rot = nutcracker.utils.rotate.rotation_based_on_quaternion(img_1,quaternion)

        quaternion_conj = condor.utils.rotation.quat_conj(quaternion)

        img_1_rot_back = nutcracker.utils.rotate.rotation_based_on_quaternion(img_1_rot,quaternion_conj)

        self.assertTrue(np.mean(np.abs(img_1 - img_1_rot_back)**2) <=  1E-3)

    def test_rotation_based_on_euler_angles(self):
        quaternion = condor.utils.rotation.rand_quat()
        quaternion_conj = condor.utils.rotation.quat_conj(quaternion)

        euler_angles = condor.utils.rotation.euler_from_quat(quaternion, 'zyx')
        euler_angles_inv = condor.utils.rotation.euler_from_quat(quaternion_conj, 'xyz')
        
        img_1_rot = nutcracker.utils.rotate.rotation_based_on_euler_angles(img_1,euler_angles,order='zyx')
        img_1_rot_back = nutcracker.utils.rotate.rotation_based_on_euler_angles(img_1_rot,euler_angles_inv,order='xyz')

        self.assertTrue(np.mean(np.abs(img_1 - img_1_rot_back)**2) <= 1E-3)
        
