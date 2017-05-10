import numpy as np
import nutcracker
from scipy import ndimage
from scipy import optimize
from scipy import signal

def find_shift_between_two_models(model_1,model_2,shift_range=5,number_of_evaluations=10,rotation_angles=[0.,0.,0.],
                                  cropping_model=0,initial_guess=[0.,0.,0.], method='brute_force',full_output=False):

    """
    Find the correct shift alignment in 3D by using a different optimization algorithms to minimise the distance between the two models.

    Args:
        :model_1(float ndarray):        3d ndarray of the fixed object
        :model_2(float ndarray):        3d ndarray ot the rotatable model

    Kwargs:
        :shift_range(float):            absolute value of the range in which the brute should be applied
        :number_of_evaluations(int):    number of grid points on which the brute force optimises
        :rotation_angles(list):         set of euler angles for rotating model_2 before applying the shift
        :method(str):                   is the optimisation method which is use to minimise the difference, default = brute_force, other option fmin_l_bfgs_b
        :full_output(bool):             returns full output as a dictionary, default = False
    """

    def shifting(x,model_1,model_2):
        x0, x1, x2 = x
        #model_2 = nutcracker.utils.rotate.rotation_based_on_euler_angles(model_2, rotation_angles)
        #model_2 = ndimage.interpolation.shift(model_2, shift=(x0, x1, x2), order=0, mode='wrap')
        model_2 = np.roll(np.roll(np.roll(model_2,int(x0),axis=0), int(x1), axis=1), int(x2), axis=2)
        return np.sum(np.abs(model_1 - model_2) ** 2)

    model_2 = nutcracker.utils.rotate.rotation_based_on_euler_angles(model_2, rotation_angles)

    # cropping the model
    if cropping_model:
        model_1 = model_1[cropping_model/2:-cropping_model/2,cropping_model/2:-cropping_model/2,cropping_model/2:-cropping_model/2]
        model_2 = model_2[cropping_model/2:-cropping_model/2,cropping_model/2:-cropping_model/2,cropping_model/2:-cropping_model/2]

    args = (model_1, model_2)

    if method == 'brute_force':
        # set parameters
        r = slice(-float(shift_range),float(shift_range),2.*shift_range/number_of_evaluations)
        ranges = [r,r,r]

        # shift retrieval brute force
        shift = optimize.brute(shifting, ranges=ranges, args=args, full_output=True, finish=optimize.fmin_bfgs)
        shift = np.array(shift)

    elif method == 'fmin_l_bfgs_b':
        #parameter for fmin_l_bfgs_b
        x0 = np.array(initial_guess)

        # fmin_l_bfgs_b optimisation
        shift = optimize.fmin_l_bfgs_b(shifting, x0, args=args, approx_grad=True)
        shift = np.array(shift)

    shift_values = shift[0]
    
    if full_output:
        if method == 'brute_force':
            out = {'shift_values':shift[0],
                   'shift_fvalues':shift[1],
                   'shift_grid':shift[2],
                   'shift_jout':shift[3]}
        elif method == 'fmin_l_bfgs_b':
            out = {'shift_values':shift[0],
                   'shift_fvalues':shift[1]}
        return out
    else:
        return shift_values
