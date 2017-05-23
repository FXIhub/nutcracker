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


def split_image(image,method='random',factor=2):
    """
    Splits a 2D diffraction pattern into two. 

    Args:
        :image(float ndarray):        2 ndarray of intensities

    Kwargs:
        :method(str):                 method which should be used for splitting the data, default='random' 
        :factor(int):                 is the factor by which the image size should be divided
    """

    # checking for method
    if method == 'random':
        image_1, image_2 = _split_image(image,factor,method_is_random=True)

    elif method == 'ordered':
        image_1, image_2 = _split_image(image,factor,method_is_random=False)
        
    else:
        print 'invalid method'

    return image_1, image_2

def _split_image(image,factor,method_is_random):
    # size of the old and the new pattern
    d = image.shape[0]
    d_new = d/factor

    # checking dimension
    if len(image.shape) == 2:

        # new pattern
        im_1 = np.zeros((d_new,d_new))
        im_2 = np.zeros((d_new,d_new))

        # iterating through pattern
        for y in range(0,d-1,factor):
            for x in range(0,d-1,factor):
                
                # part of the pattern as super pixel
                sup = image[y:y+factor,x:x+factor]
                sup = sup.ravel()
                
                # apply shuffel if necessary
                if method_is_random: np.random.shuffle(sup)

                # adding the value of the sup to two new images
                for z in range(len(sup)):
                    if z%2 == 0:
                        im_1[y/factor,x/factor] = im_1[y/factor,x/factor] + sup[z]
                    else:
                        im_2[y/factor,x/factor] = im_2[y/factor,x/factor] + sup[z]

    # checking dimension
    if len(image.shape) == 3:

        # new pattern 
        im_1 = np.zeros((d_new,d_new,d_new))
        im_2 = np.zeros((d_new,d_new,d_new))

        # iterating through pattern 
        for z in range(0,d-1,factor):
            for y in range(0,d-1,factor):
                for x in range(0,d-1,factor):

                    # part of the pattern as super pixel
                    sup = image[z:z+factor,y:y+factor,x:x+factor]
                    sup = sup.ravel()

                    # apply shuffel if necessary
                    if method_is_random: np.random.shuffle(sup)
                
                    # adding the value of the sup to two new images
                    for a in range(len(sup)):
                        if a%2 == 0:
                            im_1[z/factor,y/factor,x/factor] = im_1[z/factor,y/factor,x/factor] + sup[a]
                        else:
                            im_2[z/factor,y/factor,x/factor] = im_2[z/factor,y/factor,x/factor] + sup[a]

    return im_1, im_2
