import condor
import numpy as np
import nutcracker

def fourier_shell_correlation(model_1,model_2,model_1_is_real_space=False,model_2_is_real_space=False):
    """
    Calculates the fourier shell/ring correlation between two intensity models.

    Args: 
        :model_1(float ndarray):        2d/3d ndarray of intensities or a real space object
        :model_2(float ndarray):        2d/3d ndarray of intensities or a real space object 
    
    Kwargs:
        :model_1_is_real_space(bool):   if enabled the model will be transformed into fourier space, default = False
        :model_2_is_real_space(bool):   if enabled the model will be transformed into fourier space, default = False
    """
    
    fsc_list = []

    # transfrom the input to fourier space if necessary
    if model_1_is_real_space: model_1  = np.fft.fftshift(np.fft.fftn(model_1))
    if model_2_is_real_space: model_2  = np.fft.fftshift(np.fft.fftn(model_2))

    # shape check
    if (model_1.shape != model_2.shape):
        return "shape mismatch, shapes have to be equal!"

    # distinguish between 2D and 3D input
    if len(model_1.shape) == 2:
        r = np.sqrt((model_1.shape[0]/2)**2 + (model_1.shape[1]/2)**2)
        r = r.astype(np.int)

        # iterate through the shells
        for i in range(0,r,1):
            a, b = model_1.shape[0]/2, model_1.shape[1]/2
            x, y = np.ogrid[-a:model_1.shape[0]-a, -b:model_1.shape[1]-b]
        
            # masking the shells
            mask1 = x**2 + y**2 >= i**2
            mask2 = x**2 + y**2 < (i+1)**2
            mask3 = mask1 * mask2
            fsc_list.append((model_1[mask3] * np.conjugate(model_2[mask3])).sum() / np.sqrt((np.abs(model_1[mask3])**2).sum() * (np.abs(model_2[mask3])**2).sum()))
        
        fsc_array = np.array(fsc_list)

        return fsc_array

    elif len(model_1.shape) == 3:
        r = np.sqrt((model_1.shape[0]/2)**2 + (model_1.shape[1]/2)**2 + (model_1.shape[2]/2)**2)
        r = r.astype(np.int)
        # iterate through the shells
        for i in range(0,r,1):
            a, b, c = model_1.shape[0]/2, model_1.shape[1]/2, model_1.shape[2]/2
            x, y, z = np.ogrid[-a:model_1.shape[0]-a, -b:model_1.shape[1]-b, -c:model_1.shape[2]-c]
        
            # masking the shells
            mask1 = x**2 + y**2 + z**2 >= i**2
            mask2 = x**2 + y**2 + z**2 < (i+1)**2
            mask3 = mask1 * mask2
            fsc_list.append((model_1[mask3] * np.conjugate(model_2[mask3])).sum() / np.sqrt((np.abs(model_1[mask3])**2).sum() * (np.abs(model_2[mask3])**2).sum()))
        
        fsc_array = np.array(fsc_list)

        return fsc_array

    else:
        return "invalid dimension"
