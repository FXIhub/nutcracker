import nutcracker
import os
import numpy as np
import unittest
import h5py

_data_dir = os.path.dirname(os.path.realpath(__file__)) + "/../data"

with h5py.File(_data_dir + '/test_data.h5', 'r') as f:
    img_1 = f['real'][:]
with h5py.File(_data_dir + '/test_data_rot_shift.h5', 'r') as f:
    img_2 = f['real'][:]

class TestCaseErrorMatrix(unittest.TestCase):

    def test_error_matrix(self):
        out_mulpro = nutcracker.utils.run_error_matrix.main(model1_filename=_data_dir+'/test_data.h5',
                                                            model2_filename=_data_dir+'/test_data_rot_shift.h5',
                                                            model1_dataset='real',
                                                            model2_dataset='real',
                                                            number_of_processes=2,
                                                            chunck_size=20,
                                                            number_of_evaluations=20,
                                                            order_spline_interpolation=3,
                                                            radius_radial_mask=20./2)
        error_matrix_mulpro = np.array(out_mulpro['error_matrix']).reshape((3,20,20,20))
        

        Img_1 = np.abs(np.fft.fftshift(np.fft.fftn(img_1)))**2
        Img_2 = np.abs(np.fft.fftshift(np.fft.fftn(img_2)))**2

        out_single_function = nutcracker.utils.rotate.find_rotation_between_two_models(Img_2,Img_1,method='brute_force',
                                                                                       number_of_evaluations=20,
                                                                                       full_output=True,
                                                                                       radius_radial_mask=20./2,
                                                                                       order_spline_interpolation=3)
        error_matrix_single_function = out_single_function['rotation_grid']

        self.assertTrue(np.mean(np.abs(error_matrix_mulpro - error_matrix_single_function)**2) <= 1E-7)
        
