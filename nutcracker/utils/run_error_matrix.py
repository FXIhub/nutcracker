from error_matrix_from_brute_force import ErrorMatrixBruteForce
import numpy as np


def main(model1_filename,model2_filename,model1_dataset,model2_dataset,
         number_of_processes=1,chunck_size=10,number_of_evaluations=10,
         order_spline_interpolation=3,cropping_model=None,mask=None,
         radius_radial_mask=None,search_range=np.pi/2.):

"""
Runs the ErrorMatrixBruteForce class, which gives the errror matrix as an output.

Args:
    :model1_filename(str):                location of a hdf5 file which contains the model
    :model2_filename(str):                location of a hdf5 file which contains the model
    :model1_dataset(str):                 dataset of the file
    :model2_dataset(str):                 dataset of the file

Kwargs:
    :number_of_evaluations(int):          number of grid points on which the brute force optimises, default=10
    :order_spline_interpolation(int):     the order of the spline interpolation, has to be in range 0-5, default = 3 [from scipy.org]
    :cropping_model(int):                 cropps the model by the given vaule in total, has to be an even number, default = 0
    :mask(bool ndarray):                  provide a mask to be used for the evaluation of the cost function, default = None
    :radius_radial_mask(int):             applies a radial mask to the model with given radius, default = None
    :search_range(float/list):            absolute angle in radian in which the optimisation should be done, default = np.pi/2.
"""

    get_error_matrix = ErrorMatrixBruteForce(model1_filename,
                                             model2_filename,
                                             model1_dataset,
                                             model2_dataset,
                                             number_of_processes,
                                             chunck_size,
                                             number_of_evaluations,
                                             order_spline_interpolation,
                                             cropping_model,
                                             mask,
                                             radius_radial_mask,
                                             search_range)
    get_error_matrix.run()

if __name__ == '__main__':
    main()
