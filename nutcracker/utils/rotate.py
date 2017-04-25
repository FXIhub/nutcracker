import numpy as np
import condor
from scipy import ndimage
from scipy import optimize

def rotation_matrix(angle,axis):
    """
    Calculates the rotation matrix for a given angle and rotation axis.
    
    Args:
        :angle(float):        the angle of rotation in radian
        :axis(str):           the axis of rotation as a string
    """
    if axis == 'x':
        rot_mat = np.array([[1,0,0],
                            [0,np.cos(angle),-np.sin(angle)],
                            [0,np.sin(angle),np.cos(angle)]])
    elif axis == 'y':
        rot_mat = np.array([[np.cos(angle),0,np.sin(angle)],
                            [0,1,0],
                            [-np.sin(angle),0,np.cos(angle)]])
    elif axis == 'z':
            rot_mat = np.array([[np.cos(angle),-np.sin(angle),0],
                                [np.sin(angle),np.cos(angle),0],
                                [0,0,1]])
    else:
        return 'invalid axis'
    
    return rot_mat

def rotation_based_on_quaternion(input_model,quat,order_spline_interpolation=3):
    """
    Rotate a given model based on a given quaternion by calculating the rotation matrix.

    Args:
        :input_model(float ndarray):        3d ndarray of the rotatable object
        :quat(float ndarray):               quaternion which describes the desired rotation

    Kwargs: 
        :order_spline_interpolation(int):   the order of the spline interpolation, has to be in range 0-5, default = 3 [from scipy.org]
    """
    return _rotation_of_model(input_mode, condor.utils.rotation.rotmx_from_quat(quat), order_spline_interpolation)
    
def rotation_based_on_rotation_matrix(input_model,rotation_matrix,order_spline_interpolation=3):
    """
    Rotate a given model by a given rotation matrix in 3d.

    Args:
        :input_model(float ndarray):        3d ndarray of the rotatable object
        :rotation_matrix(float ndarray):    2d ndarray of a 3x3 rotation matrix

    Kwargs:
        :order_spline_interpolation(int):   the order of the spline interpolation, has to be in range 0-5, default = 3 [from scipy.org]
    """
    return _rotation_of_model(input_model, rotation_matrix, order_spline_interpolation)

def _rotation_of_model(input_model, rot_mat, order):
    
    # defining the coordinate system
    dim = input_model.shape[0]
    ax = np.arange(dim)
    coords = np.meshgrid(ax,ax,ax)

    # stack the meshgrid to position vectors, center them around 0 by substracting dim/2
    xyz=np.vstack([coords[2].reshape(-1)-float(dim)/2,     # x coordinate, centered
                   coords[1].reshape(-1)-float(dim)/2,     # y coordinate, centered
                   coords[0].reshape(-1)-float(dim)/2])    # z coordinate, centered
    
    # checking if matrix has the right size
    if (rot_mat.shape[0] != 3) and (rot_mat.shape[1] != 3):
        return 'invalid matrix size!'

    # rotate the coordinate system
    rot_xyz = np.dot(rot_mat,xyz)

    # extract coordinates
    x=rot_xyz[2,:]+float(dim)/2
    y=rot_xyz[1,:]+float(dim)/2
    z=rot_xyz[0,:]+float(dim)/2

    # reshaping coordinates
    x=x.reshape((dim,dim,dim))
    y=y.reshape((dim,dim,dim))
    z=z.reshape((dim,dim,dim))

    # rearange the order of the coordinates
    new_xyz=[y,x,z]

    # rotate object
    rotated_model = ndimage.interpolation.map_coordinates(input_model,new_xyz, mode='reflect', order=order)

    return rotated_model

def find_rotation_between_two_models(model_1,model_2,number_of_evaluations,
                                     full_output=False,model_1_is_intensity=True,model_2_is_intensity=True,
                                     order_spline_interpolation=3,cropping_model=0):
    """
    Finding the right alignment by rotating one model on base of a rotation matrix and using the brute force algorithm to minimise the difference between the two models.

    Args:
        :model_1(float ndarray):            3d ndarray of the fixed object                                                                                             
        :model_2(float ndarray):            3d ndarray of the rotatable object
        :number_of_evaluation(int):         number of grid points on which the brute force optimises

    Kwargs:
        :full_output(bool):                 returns full output as a dictionary, default = False
        :model_1_is_intensity(bool):        applys a fourier transformation and takes the absolute values if False, default = True
        :model_2_is_intensity(bool):        applys a fourier transformation and takes the absolute values if False, default = True
        :shift_range(int):                  absolute value of the range in which the shift part of the brute force algorithm should calculate, default = 3
        :order_spline_interpolation(int):   the order of the spline interpolation, has to be in range 0-5, default = 3 [from scipy.org]
        :cropping_model(int):               cropps the model by the given vaule in total, has to be an even number, default = 0
    """
    
    def rotation(angles,model_1,model_2):
        theta, phi, psi = angles        
        r_x = rotation_matrix(theta,'x')
        r_y = rotation_matrix(phi,'y')
        r_z = rotation_matrix(psi,'z')
        rot_mat = np.dot(np.dot(r_z,r_y),r_x)
        model_2 = rotation_based_on_rotation_matrix(model_2,rot_mat,order_spline_interpolation)
        return np.sum(np.abs(model_1 - model_2)**2)

    # cropping the model
    if cropping_model:
        model_1 = model_1[cropping_model/2:-cropping_model/2,cropping_model/2:-cropping_model/2,cropping_model/2:-cropping_model/2]
        model_2 = model_2[cropping_model/2:-cropping_model/2,cropping_model/2:-cropping_model/2,cropping_model/2:-cropping_model/2]

    # normalisation
    model_1 = model_1 * 1/(np.max(model_1))
    model_2 = model_2 * 1/(np.max(model_2))
    
    # apply FT if necessary

    if model_1_is_intensity == False: model_1 = np.abs(np.fft.fftshift(np.fft.fftn(model_1)))**2
    if model_2_is_intensity == False: model_2 = np.abs(np.fft.fftshift(np.fft.fftn(model_2)))**2
    
    # parameters for brute force optimisation
    ranges = [slice(0,2*np.pi,2*np.pi/number_of_evaluations),slice(0,2*np.pi,2*np.pi/number_of_evaluations),slice(0,2*np.pi,2*np.pi/number_of_evaluations)]
    args = (model_1,model_2)

    # brute force rotation optimisation
    rot = optimize.brute(rotation, ranges=ranges, args=args, full_output=True, finish=optimize.fmin_bfgs)
    rot = np.array(rot)
    angles = rot[0]

    if full_output:
        out = {'rotation_angles':rot[0],
               'rotation_function_values':rot[1],
               'rotation_grid':rot[2],
               'rotation_jout':rot[3],
               'rotated_model':model_2}
        return out
    else:
        return angles


def find_shift_between_two_models(model_1,model_2,shift_range,number_of_evaluations,full_output=False):
    """
    Find the right shift alignment in 3D by using a brute force algorithm to minimise the difference between the two models.

    Args:
        :model_1(float ndarray):        3d ndarray of the fixed object
        :model_2(float ndarray):        3d ndarray ot the rotatable model
        :shift_range(float):            absolute value of the range in which the brute should be applied
        :number_of_evaluations(int):    number of grid points on which the brute force optimises

    Kwargs:
        :full_output(bool):                 returns full output as a dictionary, default = False
    """

    def shifting(x,model_1,model_2):
        x0, x1, x2 = x
        model_2 = ndimage.interpolation.shift(model_2, shift=(x0, x1, x2), mode='wrap')
        return np.sum(np.abs(model_1 - model_2) ** 2)    

    # set parameters
    ranges = [slice(-shift_range,shift_range,shift_range/number_of_evaluations), slice(-shift_range,shift_range,shift_range/number_of_evaluations), slice(-shift_range,shift_range,shift_range/number_of_evaluations)]
    args = (model_1, model_2)

    # shift retrieval brute force                                                                                                                                                                                                
    shift = optimize.brute(shifting, ranges=ranges, args=args, full_output=True, finish=optimize.fmin_bfgs)                                                                                                                      
    shift = np.array(shift)                                                                                                                                                                                                      
    
    shift_values = np.array((shift[0]))

    if full_output:                                                                                                                                                                                                              
        out = {'shift_values':shift[0],                                                                                                                                                                                         
               'shift_fvalues':shift[1],                                                                                                                                                                                        
               'shift_grid':shift[2],                                                                                                                                                                                           
               'shift_jout':shift[3]}                                                                                                                                                                                           
        return out                                                                                                                                                                                                      
    else:                                                                                                                                                                                                                       
        return shift_values




# euler angles seems hopeless in this case
"""
def find_rotation_between_two_models(model_1,model_2,full_output=False,model_1_is_intensity=True,model_2_is_intensity=True,shift_range=3,extrinsic_rotation=False):
    
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
"""
