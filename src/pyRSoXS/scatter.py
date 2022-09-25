import warnings
from rotation import einsum

#Try importing cupy to replace numpy. Fall back to numpy if cupy isn't installed
try:
    import cupy as np
    MACHINE_HAS_CUDA = True
except ModuleNotFoundError:
    import numpy as np
    warnings.warn('No cupy installation found. Falling back to numpy and numba for calculations')
    MACHINE_HAS_CUDA = False



def calculate_polarization(n_iso, n_rotated, S, Evec):
    '''
    input:
        n_iso - isotropic optical tensor (3,3)
        n_rotated - rotated anisotropic optical tensors (NumZ, NumY, NumX, 3, 3)
        S - Alignment magnitude matrix (NumZ, NumY, NumX)
        Evec - Electric Field Vector (3,)
    
    output:
        P - Induced polarization (NumZ, NumY, NumX, 3)
    '''

    I = np.identity(3)
    Palign = (einsum(n_rotated,n_rotated) - I)@Evec
    Piso = (n_iso@n_iso - I)@Evec
    S = np.expand_dims(S,axis=-1)
    Piso = np.expand_dims(Piso,axis=(0,1,2))
    P = 1/4/np.pi*(S*Palign + (1-S)*Piso)
    return P


def calculate_farfield(P, k, PhysSize):
    '''
    input:
        P - Induced polarization in Real Space (NumZ, NumY, NumX, 3)
        k - x-ray propagation vector (3,)
    
    output:
        Iscatter - Scattering intensity which is the magnitude squared of 
                    the far-field projection of the electric field
    '''

    # Fast Fourier Transform of P
    Pq = np.fft.fftn(P,axes=(0,1,2))

    kmag = np.sqrt(np.sum(k**2))
    # Calculate Scattering Vectors
    qz = 2*np.pi*np.fft.fftshift(np.fft.fftfreq(Pq.shape[0],d=PhysSize))
    qy = 2*np.pi*np.fft.fftshift(np.fft.fftfreq(Pq.shape[1],d=PhysSize))
    qx = 2*np.pi*np.fft.fftshift(np.fft.fftfreq(Pq.shape[2],d=PhysSize))
    

    qzz, qyy, qxx = np.meshgrid(qz, qy, qx, indexing='ij')
    kxqx = k[0] + qxx
    kyqy = k[1] + qyy
    kzqz = k[2] + qzz

    ### Calculate the quantity k**2*(I-rr)@Pq ###
    escat_x = (Pq[...,0]*(kmag**2-kxqx**2) 
                - Pq[...,1]*kxqx*kyqy 
                - Pq[...,2]*kxqx*kzqz)

    escat_y = (Pq[...,1]*(kmag**2 - kyqy**2)
                - Pq[...,0]*kxqx*kyqy
                - Pq[...,2]*kyqy*kzqz)

    escat_z = (Pq[...,2]*(kmag**2 - kzqz**2)
                - Pq[...,0]*kxqx*kyqy
                - Pq[...,1]*kyqy*kzqz)

    if len(qz) == 1:
        # if len(qz) == 1 the morphology is 2D and we don't need to interpolate to the Ewald sphere
        Iscatter = np.abs(escat_x[0,...])**2 + np.abs(escat_y[0,...])**2 + np.abs(escat_z[0,...])**2
    else:
        # 3D morphology- interpolate onto the Ewald sphere
        Iscatter = ewald_interpolation(qx, qy, qz, escat_x, escat_y, escat_z, kmag)

    return Iscatter


def ewald_interpolation(qx, qy, qz, escat_x, escat_y, escat_z, kmag):
    '''
    Elastic scattering needs to satisfy (kout - kin)**2 = qx**2 + qy**2 + qz**2
    Given a qx and qy, find the correct qz value and interpolate the correct Escatter values. 
    Only works for 90 degree incidence currently

    input:
        qx - x-component of the scattering vector (NumX,)
        qy - y-component of the scattering vector (NumY,)
        qz - z-component of the scattering vector (NumZ,)
        escat_x, escat_y, escat_z - x, y, z components of the far-field projection (NumZ, NumY, NumX, 3)
        kmag - scattering vector magnitude (scalar)
    
    output:
        Iscatter - Scattering intensity, interpolated onto the Ewald Sphere (NumY, NumX)
    '''
    NumX = len(qx)
    NumY = len(qy)
    Iscatter = np.zeros((NumY,NumX))
    for i in range(NumY):
        for j in range(NumX):
            tmp_qy = qy[i]
            tmp_qx = qx[j]
            sqr_qz = kmag**2 - tmp_qx**2 - tmp_qy**2
            if sqr_qz < 0:
                Iscatter[i,j] = np.nan
            else:
                tmp_qz = np.sqrt(sqr_qz)

                tmp_Ex = np.interp(tmp_qz, qz, escat_x[:,i,j])
                tmp_Ey = np.interp(tmp_qz, qz, escat_y[:,i,j])
                tmp_Ez = np.interp(tmp_qz, qz, escat_z[:,i,j])

                Iscatter[i,j] = np.abs(tmp_Ex)**2 + np.abs(tmp_Ey)**2 + np.abs(tmp_Ez)**2
    
    return Iscatter
