import numpy as np
import spimage

def phase_retieval_transfer_function(images,support,full_output=False,mask=None):
    """
    Calculates the phase retrieval transfer function by using the libspimage functions.

    Args:
        :images(float ndarray):        4d ndarray of real space images
        :support(bool ndarray):        4d ndarray of the support

    Kwargs:
        :full_output(bool):            gives the full output as a dictionary
        :mask(int ndarray):            provide a mask to be used for the radial mean function, default = None
    """

    # calulating the PRTF
    output_prtf = spimage.prtf(images, support, enantio=True, translate=True)
    prtf_3d = output_prtf['prtf']

    if mask is None:
        # preparing the mask for the radial average
        nx, ny, nz = prtf_3d.shape[2], prtf_3d.shape[1], prtf_3d.shape[0]
        xx,yy,zz = np.meshgrid(np.arange(nx),np.arange(ny),np.arange(nz))
        mask = np.sqrt((xx-nx/2)**2 + (yy-ny/2)**2 + (zz-nz/2)**2) < nx/2
    
    
    prtf_centers, prtf_radial = spimage.radialMeanImage(prtf_3d, msk=mask, output_r=True)
    
    if full_output:
        out = {'prtf_3D_volume':prtf_3d,
               'prtf_radial':prtf_radial,
               'prtf_centers':prtf_centers}
        return out
    else:
        return prtf_3d
    
    
