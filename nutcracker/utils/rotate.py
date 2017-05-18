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
    return _rotation_of_model(input_model, condor.utils.rotation.rotmx_from_quat(quat), order_spline_interpolation)
    
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

def rotation_based_on_euler_angles(input_model,angles,order='zyx', order_spline_interpolation=3):
    """
    Rotate a given model by a given set of euler angles

    Args:
        :input_model(float ndarray):        3d ndarray of the rotatable object
        :angles(float ndarray):             1d ndarray of euler angles

    Kwargs:
        :order(str):                        order in which rotation matrix is constructed from th euler angles
        :order_spline_interpolation(int):   the order of the spline interpolation, has to be in range 0-5, default = 3 [from scipy.org]
    """

    r_1 = rotation_matrix(angles[0],order[0])
    r_2 = rotation_matrix(angles[1],order[1])
    r_3 = rotation_matrix(angles[2],order[2])

    rotmat = np.dot(np.dot(r_1,r_2),r_3)
    return _rotation_of_model(input_model, rotmat, order_spline_interpolation)

def _rotation_of_model(input_model, rot_mat, order):
    """
    This code is based on [http://stackoverflow.com/questions/40524490/transforming-and-resampling-a-3d-volume-with-numpy-scipy, 27.04.2017,14:47]
    """
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
    rotated_model = ndimage.interpolation.map_coordinates(input_model,new_xyz, mode='constant', order=order)

    #go back to original order (z,y,x)
    #rotated_model = np.moveaxis(rotated_model,2,0)
    
    return rotated_model

def find_rotation_between_two_models(model_1,model_2,number_of_evaluations=10,full_output=False,
                                     order_spline_interpolation=3,cropping_model=0, mask=None,
                                     method='brute_force',initial_guess=[0.,0.,0.],
                                     radius_radial_mask=np.inf,search_range=np.pi/2.,
                                     log_model=True):
    """
    Finding the correct alignment by rotating one model on base of a rotation matrix and using different optimization algorithms to minimise the distance between the two models.

    Args:
        :model_1(float ndarray):            3d ndarray of the fixed intensity object                                                                                             
        :model_2(float ndarray):            3d ndarray of the rotatable intensity object

    Kwargs:
        :number_of_evaluation(int):         number of grid points on which the brute force optimises, default = 10
        :full_output(bool):                 returns full output as a dictionary, default = False
        :order_spline_interpolation(int):   the order of the spline interpolation, has to be in range 0-5, default = 3 [from scipy.org]
        :cropping_model(int):               cropps the model by the given vaule in total, has to be an even number, default = 0
        :mask(bool ndarray):                provide a mask to be used for the evaluation of the cost function, default = None
        :method(str):                       is the optimisation method which is use to minimise the difference, default = brute_force, other option fmin_l_bfgs_b
        :initial_guess(list):               is the initila guess for the fmin_l_bfgs_b optimisation
        :radius_radial_mask(int):           applies a radial mask to the model with given radius, default = 0
        :searche_range(float/list):         absolute angle in radian in which the optimisation should be done, default = np.pi/2.
        :log_model(bool):                   if enabled it will take the logarithmic values of the models, default = True
    """    
    def costfunc(angles,model_1,model_2,mask):
        rot_mat = get_rot_matrix(angles)
        model_2 = rotation_based_on_rotation_matrix(model_2,rot_mat,order_spline_interpolation)
        return np.sum(np.abs(model_1[mask] - model_2[mask])**2)

    def get_rot_matrix(angles):
        theta, phi, psi = angles
        r_x = rotation_matrix(theta,'x')
        r_y = rotation_matrix(phi,'y')
        r_z = rotation_matrix(psi,'z')
        return np.dot(np.dot(r_z,r_y),r_x)

    # Mask
    if mask is None:
        mask = np.ones_like(model_1).astype(np.bool)
    
    # cropping the model
    if cropping_model:
        model_1 = model_1[cropping_model/2:-cropping_model/2,cropping_model/2:-cropping_model/2,cropping_model/2:-cropping_model/2]
        model_2 = model_2[cropping_model/2:-cropping_model/2,cropping_model/2:-cropping_model/2,cropping_model/2:-cropping_model/2]
        mask = mask[cropping_model/2:-cropping_model/2,cropping_model/2:-cropping_model/2,cropping_model/2:-cropping_model/2]

    # radial mask
    if radius_radial_mask:
        a, b, c = model_1.shape[0]/2, model_1.shape[1]/2, model_1.shape[2]/2
        x, y, z = np.ogrid[-a:model_1.shape[0]-a, -b:model_1.shape[1]-b, -c:model_1.shape[2]-c]
        mask_rad = np.sqrt(x**2 + y**2 + z**2) <= radius_radial_mask
        mask = mask & mask_rad
        
    # calculating the log if necessary
    if log_model:
        model_1 = np.log(model_1+1)
        model_2 = np.log(model_2+1)

    # normalisation
    model_1 = model_1 * 1/(np.max(model_1))
    model_2 = model_2 * 1/(np.max(model_2))

    # parameter for optimisation
    args = (model_1,model_2, mask)

    if method == 'brute_force':
        # parameters for brute force optimisation
        if type(search_range) == float:
            angle_range_theta = slice(initial_guess[0]-search_range,initial_guess[0]+search_range,2*search_range/number_of_evaluations)
            angle_range_phi = slice(initial_guess[1]-search_range,initial_guess[1]+search_range,2*search_range/number_of_evaluations)
            angle_range_psi = slice(initial_guess[2]-search_range,initial_guess[2]+search_range,2*search_range/number_of_evaluations)
            ranges = [angle_range_theta, angle_range_phi, angle_range_psi]
            
        if type(search_range) == list:
            angle_range_theta = slice(initial_guess[0]-search_range[0],initial_guess[0]+search_range[0],2*search_range[0]/number_of_evaluations)
            angle_range_phi = slice(initial_guess[1]-search_range[1],initial_guess[1]+search_range[1],2*search_range[1]/number_of_evaluations)
            angle_range_psi = slice(initial_guess[2]-search_range[2],initial_guess[2]+search_range[2],2*search_range[2]/number_of_evaluations)
            ranges = [angle_range_theta, angle_range_phi, angle_range_psi]

        # brute force rotation optimisation
        rot = optimize.brute(costfunc, ranges=ranges, args=args, full_output=True, finish=optimize.fmin_bfgs)
        rot = np.array(rot)
        
    if method == 'fmin_l_bfgs_b':
        #parameter for fmin_l_bfgs_b
        x0 = np.array(initial_guess)
        
        # fmin_l_bfgs_b optimisation
        rot = optimize.fmin_l_bfgs_b(costfunc, x0, args=args, approx_grad=True)
        rot = np.array(rot)

    if method == 'differential_evolution':
        bound = [(0,np.pi),(0,np.pi),(0,np.pi)]
        rot = optimize.differential_evolution(func=costfunc, bounds=bounds, args=args, strategy='best1bin', polish=True)
        
    angles = rot[0]

    # Get rotation matrix which translates model 2 into model 1
    res_rot_mat = get_rot_matrix(angles)
    model_2_rotated = rotation_based_on_rotation_matrix(model_2,res_rot_mat,order_spline_interpolation)

    if full_output:
        if method == 'brute_force':
            out = {'rotation_angles':angles,
                   'rotation_function_values':rot[1],
                   'rotation_grid':rot[2],
                   'rotation_jout':rot[3],
                   'rotated_model':model_2_rotated,
                   'rotation_matrix':res_rot_mat,
                   'mask':mask}

        if method == 'fmin_l_bfgs_b':
            out = {'rotation_angles':angles,
                   'rotation_function_values':rot[1],
                   'warnflag':rot[2]['warnflag'],
                   'gradient':rot[2]['grad'],
                   'function_calls':rot[2]['funcalls'],
                   'iterations':rot[2]['nit'],
                   'rotated_model':model_2_rotated,
                   'rotation_matrix':res_rot_mat,
                   'mask':mask}

        if method == 'differential_evolution':
            out = {'rotation_angles':angles,
                   'success':rot[1],
                   'message':rot[2]}
        return out
    else:
        return angles


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
