import numpy as np
import nutcracker
from scipy import ndimage
from scipy import optimize
from scipy import signal
import scipy as sp
"""
All functions base on the corresponding libspimage functions and are extentions to 3D.
"""

def find_center(img, msk, method=None, errout=False, **kwargs):
    """
    Find the center of a diffraction pattern.
    usage:
    ======
    x,y = find_center(img, msk)
    x,y = find_center(img, msk, method='quadrant',  x0=0, y0=0, dmax=None, threshold=None, solver='L-BFGS-B')
    x,y = find_center(img, msk, method='blurred', x0=0, y0=0, threshold=None, blur_radius=4, dmax=5)
    x,y = find_center(img, msk, method='pixelwise_fast', x0=0, y0=0, dmax=5, rmax=None)
    x,y = find_center(img, msk, method='pixelwise_slow', x0=0, y0=0, dmax=5, rmax=None)
    """

    # Default method for center finding
    if method is None: method = 'octant'

    # Find center using "octant" method
    if method == 'octant':
        x,y,z,e = find_center_octant(img, msk, **kwargs)
    # Find center using slow implementation of "pixelwise" method
    elif method == 'pixelwise':
        x,y,z,e = find_center_pixelwise(img, msk, **kwargs)
    # Return 0,0 if method is not defined
    else:
        x,y,z,e = (0,0,0,0)
        print "There is no center finding method %s" %method

    # Check for reasonable numbers
    if abs(x) > img.shape[2]/2: x = 0
    if abs(y) > img.shape[1]/2: y = 0
    if abs(z) > img.shape[0]/2: z = 0

    if errout:
        return (x,y,z,e)
    else:
        return (x,y,z)

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
    img_new[y_start:y_stop,z_start:z_stop,x_start:x_stop] = np.rot90(np.rot90(np.rot90(img[z_start:z_stop,y_start:y_stop,x_start:x_stop])))
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



def find_center_octant(img, msk, dmax=5, x0=0, y0=0, z0=0, threshold=None, solver='L-BFGS-B'):
    """
    Find the center of a diffraction volume using the octant method.
    For every possible center shift (within - dmax ... + dmax) a centrosymmetric mask is calculated.
    The image is divided into eight octants A,B,C,D,E,F,G,H around any possible center position:
        +----+----+
       / E  /  F /|
      +----+----+ |
     /  A / B  /| |
    +----+----+ |/|
    | A  | B  |/|H|
    +----+----| |/
    | C  | D  |/ 
    +----+----+

    Depending of different center positions, the cost function

    E = |sum(w_A*A)-sum(w_H*H)|^2 + |sum(w_B*B) - sum(w_G*G)|^2 + |sum(w_F*F)-sum(w_C*C)|^2 + |sum(w_E*E)-sum(wD*D)|) / sum(wA + wB + wC + wD + wE + wF + wG + wH)

    with w_i being the respective centrosymmetric masks for i=A,B,C,D,E,F,G,H
    is minimized using a given solver (default = 'L-BFGS-B').
    usage: 
    ======
    x,y,z = find_center_quadrant(img, msk,  x0=0, y0=0, z0=0,  dmax=None, threshold=None, solver='L-BFGS-B')
    """    

    class CentroSymmetricMask:
        def __init__(self, mask, dx, dy, dz):
            mask = mask.astype(np.bool)
            self.omask = mask
            self.nz, self.ny, self.nx = mask.shape
            self.dx = dx
            self.dy = dy
            self.dz = dz
            self.define_mask()

        def define_mask(self):
            csmask0 = sp.flipud(sp.fliplr(self.omask))
            csmask = np.zeros((self.nz+4*self.dz, self.ny+4*self.dy, self.nx+4*self.dx)).astype(np.bool)
            csmask[self.dz:self.dz+self.nz,self.dy:self.dy+self.ny,self.dx:self.dx+self.nx] = csmask0
            self.csmask = csmask

        def get(self, x0=0, y0=0, z=0):
            return self.omask & self.csmask[self.dz-2*z0:self.dz+self.nz-2*z0,self.dy-2*y0:self.dy+self.ny-2*y0,self.dx-2*x0:self.dx+self.nx-2*x0]

    class Minimizer:
        def __init__(self,img, msk, x0, y0, z0, maxshift, solver):
            self.csmask = CentroSymmetricMask(msk, 2*maxshift, 2*maxshift, 2*maxshift)
            self.image = img
            self.Nz, self.Ny, self.Nx = img.shape
            self.x0_initial = x0
            self.y0_initial = y0
            self.z0_initial = z0
            self.maxshift = maxshift
            self.solver = solver

        def manipulate(self, threshold=None):
            if threshold is not None:
                self.image[self.image < threshold] = 0.
            #self.image = np.log(self.image)
            
        def update_center(self, x0,y0,z0):
            self.zt = max(2*z0,0) 
            self.zb = min(self.Nz + 2*z0, self.Nz)
            self.zm = (self.zt + self.zb)/2
            self.yah = max(2*y0, 0)
            self.ybh = min(self.Ny + 2*y0, self.Ny)
            self.ym = (self.yah + self.ybh)/2
            self.xl = max(2*x0, 0)
            self.xr = min(self.Nx + 2*x0, self.Nx)
            self.xm = (self.xl + self.xr)/2

        def update_mask(self, x0,y0,z0):
            self.mask = self.csmask.get(x0,y0,z0)

        def error(self, p):
            if np.alltrue(np.isnan(p)):
                pass
            else:
                [x0, y0, z0] = p
                self.update_center(x0,y0,z0)
                self.update_mask(x0,y0,z0)

                wA = self.mask[self.zt:self.zm, self.ym:self.yah, self.xl:self.xm]
                wB = self.mask[self.zt:self.zm, self.ym:self.yah, self.xm:self.xr]
                wC = self.mask[self.zm:self.zb, self.ym:self.yah, self.xl:self.xm]
                wD = self.mask[self.zm:self.zb, self.ym:self.yah, self.xm:self.xr]
                wE = self.mask[self.zt:self.zm, self.ybh:self.ym, self.xl:self.xm]
                wF = self.mask[self.zt:self.zm, self.ybh:self.ym, self.xm:self.xr]
                wG = self.mask[self.zm:self.zb, self.ybh:self.ym, self.xl:self.xm]
                wH = self.mask[self.zm:self.zb, self.ybh:self.ym, self.xm:self.xr]
                
                A = self.image[self.zt:self.zm, self.ym:self.yah, self.xl:self.xm][wA]
                B = self.image[self.zt:self.zm, self.ym:self.yah, self.xm:self.xr][wB]
                C = self.image[self.zm:self.zb, self.ym:self.yah, self.xl:self.xm][wC]
                D = self.image[self.zm:self.zb, self.ym:self.yah, self.xm:self.xr][wD]
                E = self.image[self.zt:self.zm, self.ybh:self.ym, self.xl:self.xm][wE]
                F = self.image[self.zt:self.zm, self.ybh:self.ym, self.xm:self.xr][wF]
                G = self.image[self.zm:self.zb, self.ybh:self.ym, self.xl:self.xm][wG]
                H = self.image[self.zm:self.zb, self.ybh:self.ym, self.xm:self.xr][wH]
            
                norm = 2*(wA.sum() + wB.sum())
                error = np.sqrt( abs(A.sum() - H.sum())**2 + abs(B.sum() - G.sum())**2 + abs(E.sum() - D.sum())**2 + abs(F.sum() - C.sum())**2 ) / norm
                return error
            
        def error_smooth(self, p):
            if np.alltrue(np.isnan(p)):
                pass
            else:
                [x0, y0, z0] = p
                x0f = np.floor(x0)
                x0f = np.int(x0f)
                x0c = x0f + 1
                y0f = np.floor(y0)
                y0f = np.int(y0f)
                y0c = y0f + 1
                z0f = np.floor(z0)
                z0f = np.int(z0f)
                z0c = z0f + 1
                err_fff = self.error([x0f, y0f, z0f])
                err_ccc = self.error([x0c, y0c, z0c])
                err_cff = self.error([x0c, y0f, z0f])
                err_fcc = self.error([x0f, y0c, z0c])
                err_cfc = self.error([x0c, y0f, z0c])
                err_fcf = self.error([x0f, y0c, z0f])
                err_ffc = self.error([x0f, y0f, z0c])
                err_ccf = self.error([x0c, y0c, z0f])
            
                wfff = (x0c - x0) * (y0c - y0) * (z0c - z0)
                wccc = (x0 - x0f) * (y0 - y0f) * (z0 - z0f)
                wcff = (x0 - x0f) * (y0c - y0) * (z0c - z0)
                wfcc = (x0c - x0) * (y0 - y0f) * (z0 - z0f)
                wcfc = (x0 - x0f) * (y0c - y0) * (z0 - z0f) 
                wfcf = (x0c - x0) * (y0 - y0f) * (z0c - z0)
                wffc = (x0c - x0) * (y0c - y0) * (z0 - z0f)
                wccf = (x0 - x0f) * (y0 - y0f) * (z0c - z0)
                error = wfff*err_fff + wccc*err_ccc + wcff*err_cff + wfcc*err_fcc + wcfc*err_cfc + wfcf*err_fcf + wffc*err_ffc + wccf*err_ccf
                return error

        def error_and_gradient(self,p):
            if np.alltrue(np.isnan(p)):
                pass
            else:
                err = self.error(p)
                dx = self.egrgror([p[0]+1, p[1]], p[2]) - err
                dy = self.error([p[0],     p[1]+1], p[2]) - err
                dz = self.error([p[0],     p[1], p[2]+1]) - err
                return err, np.array([dx,dy,dz])

        def start(self):
            self.res = sp.optimize.minimize(self.error_smooth, np.array([self.x0_initial,self.y0_initial,self.z0_initial]), 
                                            method=self.solver, jac=False, options={'disp':False}, 
                                            bounds=[(-self.maxshift+1, self.maxshift-1), (-self.maxshift+1, self.maxshift-1), (-self.maxshift+1, self.maxshift-1)])

        def error_landscape(self):
            XYZ = np.mgrid[-self.maxshift:self.maxshift:(2*self.maxshift+1)*1j]
            E = np.zeros((2*self.maxshift+1,2*self.maxshift+1,2*self.maxshift+1))
            for j in range(E.shape[0]):
                for i in range(E.shape[1]):
                    for k in range (E.shape[2]):
                        E[j,i,k] = self.error([XYZ[i],XYZ[j],XYZ[k]])
            return E

    m = Minimizer(img, msk, x0, y0, z0, dmax, solver)
    m.manipulate(threshold)
    m.start()
    x = m.res["x"][0]
    y = m.res["x"][1]
    z = m.res["x"][2]
    e = m.error_smooth(m.res["x"])
    return (x,y,z,e)




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
