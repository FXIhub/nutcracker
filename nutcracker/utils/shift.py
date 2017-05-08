import numpy as np
import nutcracker
from scipy import ndimage
from scipy import optimize
from scipy import signal
"""
All _functions base on the corresponding libspimage function and are extentions to 3D.
"""

def _symmetrize(M,cx,cy,cz):
    M_new = M.copy()
    M_new *= _turn180(M,cx,cy,cz)
    return M_new

def _turn180(img,cx=None,cy=None,cz=None):
    if cx == None:
        cx1 = (img.shape[2]-1)/2
    if cy == None:
        cy1 = (img.shape[1]-1)/2
    if cz == None:
        cz1 = (img.shape[0]-1)/2
    cx1 = round(cx*2)/2.
    cy1 = round(cy*2)/2.
    cz1 = round(cz*2)/2.
    Nx1 = int(2*min([cx1,img.shape[2]-1-cx1]))+1
    Ny1 = int(2*min([cy1,img.shape[1]-1-cy1]))+1
    Nz1 = int(2*min([cz1,img.shape[0]-1-cz1]))+1
    z_start = int(round(cz1-(Nz1-1)/2.))
    z_stop = int(round(cz1+(Nz1-1)/2.))+1
    y_start = int(round(cy1-(Ny1-1)/2.))
    y_stop = int(round(cy1+(Ny1-1)/2.))+1
    x_start = int(round(cx1-(Nx1-1)/2.))
    x_stop = int(round(cx1+(Nx1-1)/2.))+1
    img_new = np.zeros(shape=(img.shape[0],img.shape[1],img.shape[2]),dtype=img.dtype)
    img_new[z_start:z_stop,y_start:y_stop,x_start:x_stop] = np.rot90(np.rot90(np.rot90(img[z_start:z_stop,y_start:y_stop,x_start:x_stop])))
    return img_new

def _gaussian_smooth_3d2d1d(I,sm,precision=1.):
    N = 2*int(np.round(precision*sm))+1
    if len(I.shape) == 3:
        kernel = np.zeros(shape=(N,N))
        X,Y,Z = np.meshgrid(np.arange(0,N,1),np.arange(0,N,1),np.arange(0,N,1))
        X = X-N/2
        kernel = np.exp(X**2/(2.0*sm**2))
        kernel /= kernel.sum()
        Ism = scipy.signal.convolve(I,kernel,mode='same')
        return Ism
    if len(I.shape) == 2:
        kernel = numpy.zeros(shape=(N,N))
        X,Y = numpy.meshgrid(numpy.arange(0,N,1),numpy.arange(0,N,1))
        X = X-N/2
        kernel = numpy.exp(X**2/(2.0*sm**2))
        kernel /= kernel.sum()
        Ism = signal.convolve2d(I,kernel,mode='same',boundary='wrap')
        return Ism
    elif len(I.shape) == 1:
        print "Error input"
        return []

def find_center_pixelwise(img, msk, x0=0, y0=0, z0=0, dmax=5, rmax=None):
    """
    Find the center of an 3D image using pixelwise comparison of centro-symmetric pixels.
    This code bases on the libspimage _spimage_find_center.py function and is an extention to 3D.

    Args:
        :image(float ndarray):        3d ndarray of the image
        :mask(int ndarray):           3d ndarray of the mask

    Kwargs:
        :x0(int):
        :y0(int):
        :z0(int):
        :dmax(5):
        :rmax:
    """
    
    s = img.shape
    if rmax is None: rmax = np.sqrt(3)*max(s)

    cx_g = (s[2]-1)/2.+x0
    cy_g = (s[1]-1)/2.+y0
    cz_g = (s[0]-1)/2.+z0

    cx_g = np.round(cx_g*2)/2.
    cy_g = np.round(cy_g*2)/2.
    cz_g = np.round(cz_g*2)/2.

    ddc = 0.5
    N_sam1= int(np.round(2*dmax/ddc))+1

    cx_sam1 = np.linspace(cx_g-dmax,cx_g+dmax,N_sam1)
    cy_sam1 = np.linspace(cy_g-dmax,cy_g+dmax,N_sam1)
    cz_sam1 = np.linspace(cz_g-dmax,cz_g+dmax,N_sam1)

    N_sam2= int(np.round(4*dmax/ddc))+1

    cx_sam2 = np.linspace(cx_g-dmax*2,cx_g+dmax*2,N_sam2)
    cy_sam2 = np.linspace(cy_g-dmax*2,cy_g+dmax*2,N_sam2)
    cz_sam2 = np.linspace(cz_g-dmax*2,cz_g+dmax*2,N_sam2)

    msk_ext = msk.copy()
    for cz in cz_sam2:
        for cy in cy_sam2:
            for cx in cx_sam2:
                msk_ext *= _symmetrize(msk,cx,cy,cz)
    Nme = msk_ext.sum()
    errs = np.zeros(shape=(N_sam1,N_sam1,N_sam1))
    r_max_sq = rmax**2

    X,Y,Z = np.meshgrid(np.arange(img.shape[2]),np.arange(img.shape[1]),np.arange(img.shape[0]))    
    for cx,icx in zip(cx_sam1,np.arange(N_sam1)):
        for cy,icy in zip(cy_sam1,np.arange(N_sam1)):
            for cz,icz in zip(cz_sam1,np.arange(N_sam1)):
                r_sq = ((X-cx)**2+(Y-cy)**2+(Z-cz)**2)
                rmsk = r_sq < r_max_sq
                img_turned = _turn180(img,cx,cy,cz)
                diff = abs((img-img_turned)*msk_ext*rmsk)
                errs[icz,icy,icx] = diff.sum()

    errs_sm = _gaussian_smooth_3d2d1d(errs,dmax)
    i_min = errs.flatten().argmin()
    cxi_min = i_min % N_sam1
    cyi_min = i_min/N_sam1
    czi_min = i_min/N_sam1

    cx_r = cx_sam1[cxi_min]
    cy_r = cy_sam1[cyi_min]
    cz_r = cz_sam1[czi_min]
    x = cx_r-(s[2]-1)/2.
    y = cy_r-(s[1]-1)/2.
    z = cz_r-(s[0]-1)/2.
    return (x,y,z, errs.flatten()[i_min])

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
