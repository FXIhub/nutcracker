import nutcracker
import os
import numpy as np

_data_dir = os.path.dirname(os.path.realpath(__file__)) + "/../data"

class TestCaseErrorMatrix(unittest.TestCase):

    def test_error_matrix(self):
        error_matrix_chuncks = nutcracker.utils.run_error_matrix.main(model1_filename=_data_dir+'/test_data.h5',
                                                                      model2_filename=_data_dir+'/test_data_rot_shift.h5',
                                                                      model1_dataset='real',
                                                                      model2_dataset='real',
                                                                      number_of_processes=1,
                                                                      chunck_size=10,
                                                                      number_of_evaluations=10,
                                                                      order_spline_interpolation=3,
                                                                      cropping_model=44,
                                                                      mask=None,
                                                                      radius_radial_mask=None,
                                                                      search_range=np.pi/2.)
        
