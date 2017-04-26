import numpy as np
import spimage

# extention of the libspimage radialImage function to 3D
def radial_image_3D(image,mode="mean",cx=None,cy=None,cz=None,msk=None,output_r=False):
    """
    Calculates the radial average of a 3D volume.
    
    Args:
        :image(float ndarray):        3d ndarray of the volume

    Kwargs:
        :mode(str):                   mode of how the average should be done
        :cx(int):                     x coordinate of the image center
        :cy(int):                     y coordinate of the image center 
        :cz(int):                     z coordinate of the image center
        :output_r(bool):              gives the values and the radius
    """
    if mode == "mean": f = np.mean
    elif mode == "sum": f = np.sum
    elif mode == "std": f = np.std
    elif mode == "median": f = np.median
    else:
        logging.error("ERROR: No valid mode given for radial projection.")
        return None
    if cx is None: 
        cx = (img.shape[2]-1)/2.0
    if cy is None: 
        cy = (img.shape[1]-1)/2.0
    if cz is None: 
        cz = (img.shape[0]-1)/2.0
    X,Y,Z = np.meshgrid(np.arange(img.shape[2]),np.arange(img.shape[1]),np.arange(img.shape[0]))
    R = np.sqrt((X - float(cx))**2 + (Y - float(cy))**2 + (Z - float(cz))**2)
    R = R.round()
    if msk is not None:
        if (msk == 0).sum() > 0:
            R[msk == 0] = -1
    radii = np.arange(R.min(),R.max()+1,1)
    if radii[0] == -1:
        radii = radii[1:]
    values = np.zeros_like(radii)
    for i in range(0,len(radii)):
        tmp = R==radii[i]
        if tmp.sum() > 0:
            values[i] = f(img[tmp])
        else:
            values[i] = np.nan
    if (np.isfinite(values) == False).sum() > 0:
        tmp = np.isfinite(values)
        values = values[tmp]
        radii  = radii[tmp]
    if output_r:
        return radii,values
    else:
        return values


def phase_retieval_transfer_function(images,support,full_output=False):
    """
    Calculates the phase retrieval transfer function.

    Args:
        :images(float ndarray):        4d ndarray of real space images
        :support(bool ndarray):        4d ndarray of the support

    Kwargs:
        :full_output(bool):            gives the full output as a dictionary
    """

    # calulating the PRTF
    output_prtf = spimage.prtf(imgages, support, enantio=True, translate=True)
    prtf_3d = output_prtf['prtf']

    # preparing the mask for the radial average
    nx, ny, nz = prtf_3d.shape[2], prtf_3d.shape[1], prtf_3d.shape[0]
    xx,yy,zz = np.meshgrid(np.arange(nx),np.arange(ny),np.arange(nz))
    mask_radial = np.sqrt((xx-nx/2)**2 + (yy-ny/2)**2 + (zz-nz/2)**2) < nx/2
    
    
    prtf_centers, prtf_radial = radial_image__3D(prtf_3d, msk=mask_radial, output_r=True)
    
    if full_output:
        out = {'prtf_3D_volume':prtf_3d,
               'prtf_radial':prtf_radial,
               'prtf_centers':prtf_centers}
        return out
    else:
        return prtf_radial
    
    
