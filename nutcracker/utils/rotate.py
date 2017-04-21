import numpy as np
import condor
from scipy import ndimage
from scipy import optimize

def rotate_based_on_quaternion(input_model,quat,quat_is_extrinsic=False,extrinsic_rotation=False):
    """
    Rotate a given model based on a given quaternion by calculating the euler angles. Standart x convention is used, that means a rotation around z-x'-z'' (intrinsic). Also extrinsic rotation could be enabled, rotation around z-y-x.

    Args:
        :input_model(float ndarray):        3d ndarray of the rotatable object
        :quat(float ndarray):               quaternion which describes the desired rotation

    Kwargs: 
        :quat_is_extrinsic(bool):           quat quaternion is extrinsic and should be converted to instrinsic, default = False
        :extrinsic_rotation(bool):          if enabled the model will be rotated by extrinsic formalism, default = False
    """

    rotated_model = input_model[:]

    if extrinsic_rotation:
        #convert quaternion to extrinsic if necessary                                                                                                                                                              
        if quat_is_extrinsic == False: quat = condor.utils.rotation.quat_conj(quat)

        # normalise quat
        quat = condor.utils.rotation.norm_quat(quat)
        
        # calculating the euler angles                                                                                                                                                                             
        euler_angles = condor.utils.rotation.euler_from_quat(quat,mode='zyx')
        euler_angles = euler_angles * 180./np.pi

        # rotation around z                                                                                                                                                                                        
        rotated_model = ndimage.interpolation.rotate(rotated_model, euler_angles[0], axes=(1,2), reshape=False, mode='wrap')

        # rotation around y                                                                                                                                                                                       
        rotated_model = ndimage.interpolation.rotate(rotated_model, euler_angles[1], axes=(2,0), reshape=False, mode='wrap')

        # rotation around x                                                                                                                                                                                      
        rotated_model = ndimage.interpolation.rotate(rotated_model, euler_angles[2], axes=(0,1), reshape=False, mode='wrap')

        return rotated_model

    else:
        #convert quaternion to intrinsic if necessary
        if quat_is_extrinsic: quat = condor.utils.rotation.quat_conj(quat)
        
        # normalise quat                                                                                                                                                                                           
        quat = condor.utils.rotation.norm_quat(quat)

        # calculating the euler angles
        euler_angles = condor.utils.rotation.euler_from_quat(quat,mode='zxz')
        euler_angles = euler_angles * 180./np.pi

        # rotation around z
        rotated_model = ndimage.interpolation.rotate(rotated_model, euler_angles[0], axes=(1,2), reshape=False, mode='wrap')

        # rotation around x'
        rotated_model = ndimage.interpolation.rotate(rotated_model, euler_angles[1], axes=(0,1), reshape=False, mode='wrap')

        # rotation around z''
        rotated_model = ndimage.interpolation.rotate(rotated_model, euler_angles[2], axes=(1,2), reshape=False, mode='wrap')

        return rotated_model


def find_rotation_between_two_models(model_1,model_2,full_output=False,model_1_is_intensity=True,model_2_is_intensity=True,shift_range=3,extrinsic_rotation=False):
    """
    Finding the right alignment by rotating/shifting one model in accordance with standart x convention and using a brute force algorithm to minimise the difference between the two models. Also a extrinsic rotation formalism (rotation around z-y-x) could be enabled.

    Args:
        :model_1(float ndarray):        3d ndarray of the fixed object
        :model_2(float ndarray):        3d ndarray of the rotatable object

    Kwargs:
        :full_output(float ndarray):    returns full output as a dictionary, default = False
        :model_1_is_ft(bool):           apply a fourier transformation and takes the absolute values if False, default = True
        :model_2_is_ft(bool):           apply a fourier transformation and takes the absolute values if False, default = True
        :shift_range(int):              absolute value of the range in which the shift part of the brute force algorithm should calculate, default = 3
        :extrinsic_rotation(bool):      if enabled the model will be rotated by extrinsic formalism, default = False
    """

    # defining the functions for the brute force algorithm
    def rotation_z(a,model_1,model_2):
        model_2 = ndimage.interpolation.rotate(model_2, a, axes=(1,2), reshape=False, mode='wrap')
        return np.sum(np.abs(model_1 - model_2) ** 2)

    def rotation_y(c,model_1,model_2):
        model_2 = ndimage.interpolation.rotate(model_2, c, axes=(2,0), reshape=False, mode='wrap')
        return np.sum(np.abs(model_1 - model_2) ** 2)

    def rotation_x(b,model_1,model_2):
        model_2 = ndimage.interpolation.rotate(model_2, b, axes=(0,1), reshape=False, mode='wrap')
        return np.sum(np.abs(model_1 - model_2) ** 2)
    
    def shifting(x,model_1,model_2):
        x0, x1, x2 = x
        model_2 = ndimage.interpolation.shift(model_2, shift=(x0, x1, x2), mode='wrap')
        return np.sum(np.abs(model_1 - model_2) ** 2)

    # normalising the models
    model_1 = model_1 / model_1.max()
    model_2 = model_2 / model_2.max()
    
    # fist rough shift alignment
    coord_1 = ndimage.measurements.center_of_mass(model_1)
    coord_2 = ndimage.measurements.center_of_mass(model_2)
    model_2 = ndimage.interpolation.shift(model_2, shift=(coord_1[0] - coord_2[0], coord_1[1] - coord_2[1], coord_1[2] - coord_2[2]))

    # apply FT if necessary
    if model_1_is_intensity == False: model_1 = np.abs(np.fft.fftshift(np.fft.fftn(model_1)))
    if model_2_is_intensity == False: model_2 = np.abs(np.fft.fftshift(np.fft.fftn(model_2)))

    # parameters for brute force optimisation
    ranges = [(0,180)]
    args = (model_1,model_2)
    #Ns = 1

    if extrinsic_rotation:
        # rotation around z axis                                                                                                                                                                                   
        rot_z = optimize.brute(rotation_z, ranges=ranges, args=args, full_output=True, finish=optimize.fmin_bfgs)
        rot_z = np.array(rot_z)
        model_2 = ndimage.interpolation.rotate(model_2, rot_z[0][0], axes=(1,2), reshape=False, mode='wrap')

        # rotation around y axis                                                                                                                                                                                  
        rot_y = optimize.brute(rotation_y, ranges=ranges, args=args, full_output=True, finish=optimize.fmin_bfgs)
        rot_y = np.array(rot_y)
        model_2 = ndimage.interpolation.rotate(model_2, rot_y[0][0], axes=(2,0), reshape=False, mode='wrap')

        # rotation around x axis                                                                                                                                                                                 
        rot_x = optimize.brute(rotation_x, ranges=ranges, args=args, full_output=True, finish=optimize.fmin_bfgs)
        rot_x = np.array(rot_x)

        # shift retrieval parameters                                                                                                                                                                               
        ranges = [slice(-shift_range,shift_range,1), slice(-shift_range,shift_range,1), slice(-shift_range,shift_range,1)]
        args = (model_1, model_2)

        # shift retrieval brute force                                                                                                                                                                              
        shift = optimize.brute(shifting, ranges=ranges, args=args, full_output=True, finish=optimize.fmin_bfgs)
        shift = np.array(shift)

        angles = np.array((rot_z[0][0],rot_y[0][0],rot_x[0][0]))
        shift_values = np.array((shift[0]))

        if full_output:
            out = {'z_angle':rot_z[0],
                   'z_fvalue':rot_z[1],
                   'z_grid':rot_z[2],
                   'z_jout':rot_z[3],
                   'x_angle':rot_x[0],
                   'x_fvalue':rot_x[1],
                   'x_grid':rot_x[2],
                   'x_jout':rot_x[3],
                   'z_2_angle':rot_z_2[0],
                   'z_2_fvalue':rot_z_2[1],
                   'z_2_grid':rot_z_2[2],
                   'z_2_jout':rot_z_2[3],
                   'shift_values':shift[0],
                   'shift_fvalues':shift[1],
                   'shift_grid':shift[2],
                   'shift_jout':shift[3]}
            return angles, out
        else:
            return angles, shift_values


    else:
        # rotation around z axis
        rot_z = optimize.brute(rotation_z, ranges=ranges, args=args, full_output=True, finish=optimize.fmin_bfgs)
        rot_z = np.array(rot_z)
        model_2 = ndimage.interpolation.rotate(model_2, rot_z[0][0], axes=(1,2), reshape=False, mode='wrap')
    
        # rotation around x' axis
        rot_x = optimize.brute(rotation_x, ranges=ranges, args=args, full_output=True, finish=optimize.fmin_bfgs)
        rot_x = np.array(rot_x)
        model_2 = ndimage.interpolation.rotate(model_2, rot_x[0][0], axes=(0,1), reshape=False, mode='wrap')

        # rotation around z'' axis
        rot_z_2 = optimize.brute(rotation_z, ranges=ranges, args=args, full_output=True, finish=optimize.fmin_bfgs)
        rot_z_2 = np.array(rot_z_2)
    
        # shift retrieval parameters
        ranges = [slice(-shift_range,shift_range,1), slice(-shift_range,shift_range,1), slice(-shift_range,shift_range,1)]
        args = (model_1, model_2)
    
        # shift retrieval brute force
        shift = optimize.brute(shifting, ranges=ranges, args=args, full_output=True, finish=optimize.fmin_bfgs)
        shift = np.array(shift)

        angles = np.array((rot_z[0][0],rot_x[0][0],rot_z_2[0][0]))
        shift_values = np.array((shift[0])) 

        if full_output:
            out = {'z_angle':rot_z[0],
                   'z_fvalue':rot_z[1],
                   'z_grid':rot_z[2],
                   'z_jout':rot_z[3],
                   'x_angle':rot_x[0],
                   'x_fvalue':rot_x[1],
                   'x_grid':rot_x[2],
                   'x_jout':rot_x[3],
                   'z_2_angle':rot_z_2[0],
                   'z_2_fvalue':rot_z_2[1],
                   'z_2_grid':rot_z_2[2],
                   'z_2_jout':rot_z_2[3],
                   'shift_values':shift[0],
                   'shift_fvalues':shift[1],
                   'shift_grid':shift[2],
                   'shift_jout':shift[3]}
            return angles, out
        else:
            return angles, shift_values
