import h5py
import pathlib
import numba as nb
import warnings

### Try importing cupy to replace numpy. Fall back to numpy if cupy isn't installed
try:
    import cupy as np
    nb_compile = False
except ModuleNotFoundError:
    import numpy as np
    warnings.warn('No cupy installation found. Falling back to numpy for calculations')
    nb_compile = True


def numba_compile(func):
    if nb_compile:
        func = nb.njit(func,parallel=True)
    else:
        pass
    return func

@numba_compile
def nb_einsum(array1, array2):
    '''
    numba compiled and optimized function that replaces 
    np.einsum for the operation '...ij,...jk->...ik'

    input:
        array1: n-dimensional array where the last two dimensions are (3,3)
        array2: n-dimensional array where the last two dimensions are (3,3)
    
    output:
        result: n-dimensional array where the last two dimensions are (3,3)
    '''

    shape = array1.shape
    array1 = np.reshape(array1,(-1,3,3))
    array2 = np.reshape(array2,(-1,3,3))
    result = np.empty(array1.shape)

    for s in nb.prange(shape[0]):
        for i in range(3):
            for k in range(3):
                acc = 0
                for j in range(3):
                    acc += array1[s,i,j]*array2[s,j,k]
                result[s,i,k] = acc

    result = np.reshape(result,shape)
    return result

@numba_compile
def create_Rzyz(phi, theta, psi):
    '''
    Creates the full set of 3x3 ZYZ rotation matrices from Euler Angles
    
    input:
        phi: 1st rotation, about Z-axis (NumZ, NumY, NumX)
        theta: 2nd rotation, about Y-axis (NumZ, NumY, NumX)
        psi: 3rd rotation, about Z-axis (NumZ, NumY, NumX)
    
    output:
        r_zyz: ZYZ rotation matrices (NumZ, NumY, NumX, 3, 3)
    '''

    rz1 = create_Rz(phi)
    ry = create_Ry(theta)
    rz2 = create_Rz(psi)
    r_zyz = nb_einsum(rz2,nb_einsum(ry,rz1))
    return r_zyz

@numba_compile
def create_Rz(angles):
    rz = np.zeros((*angles.shape,3,3))
    tmp_cos = np.cos(angles)
    tmp_sin = np.sin(angles)
    rz[...,0,0] = tmp_cos
    rz[...,1,1] = tmp_cos
    rz[...,1,0] = tmp_sin
    rz[...,0,1] = -tmp_sin
    rz[...,2,2] = 1
    return rz

@numba_compile
def create_Ry(angles):
    ry = np.zeros((*angles.shape,3,3))
    tmp_cos = np.cos(angles)
    tmp_sin = np.sin(angles)
    ry[...,0,0] = tmp_cos
    ry[...,2,2] = tmp_cos
    ry[...,2,0] = -tmp_sin
    ry[...,0,2] = tmp_sin
    ry[...,1,1] = 1
    return ry

@numba_compile
def create_Rx(angles):
    rx = np.zeros((*angles.shape,3,3))
    tmp_cos = np.cos(angles)
    tmp_sin = np.sin(angles)
    rx[...,0,0] = 1
    rx[...,1,1] = tmp_cos
    rx[...,2,2] = tmp_cos
    rx[...,2,1] = tmp_sin
    rx[...,1,2] = -tmp_sin
    return rx

@numba_compile
def rotate_n(n,r_zyz):
    '''
    input:
        n - aligned optical tensor (3,3)
        r_zyz - Rotation matrices constructed using Euler angles in ZYZ ordering (NumZ, NumY, NumX, 3, 3)

    output:
        n_rotated - rotated optical tensors (NumZ, NumY, NumX, 3, 3)
    '''
    shape = rotmat.shape
    rotmat = np.reshape(rotmat,(-1,3,3))
    result1 = np.empty(rotmat.shape,dtype=complex)
    n_rotated = result1.copy()

    for s in nb.prange(shape[0]):
        for i in range(3):
            for k in range(3):
                acc = 0
                for j in range(3):
                    acc += n[i,j]*rotmat[s,k,j]
                result1[s,i,k] = acc
    
    for s in nb.prange(shape[0]):
        for i in range(3):
            for k in range(3):
                acc = 0
                for j in range(3):
                    acc += rotmat[s,i,j]*result1[s,j,k]
                n_rotated[s,i,k] = acc
    
    n_rotated = np.reshape(n_rotated,shape)
    return n_rotated



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
    Palign = (nb_einsum(n_rotated,n_rotated) - I)@Evec
    Piso = (n_iso@n_iso - I)@Evec
    S = np.expand_dims(S,axis=-1)
    Piso = np.expand_dims(Piso,axis=(0,1,2))
    P = 1/4/np.pi*(S*Palign + (1-S)*Piso)
    return P

def calculate_farfield(Pq, k, PhysSize):
    '''
    input:
        Pq - Induced polarization in Fourier Space (NumZ, NumY, NumX, 3)
        k - x-ray propagation vector (3,)
    
    output:
        Iscatter - Scattering intensity which is the magnitude squared of 
                    the far-field projection of the electric field
    '''

    kmag = np.sqrt(np.sum(k**2))
    # Calculate Scattering Vectors
    qz = 2*np.pi*np.fft.fftshift(np.fft.fftfreq(Pq.shape[0],d=PhysSize))
    qy = 2*np.pi*np.fft.fftshift(np.fft.fftfreq(Pq.shape[1],d=PhysSize))
    qx = 2*np.pi*np.fft.fftshift(np.fft.fftfreq(Pq.shape[2],d=PhysSize))
    

    qzz, qyy, qxx = np.meshgrid(qz, qy, qx, indexing='ij')
    kxqx = k[0] - qxx
    kyqy = k[1] - qyy
    kzqz = k[2] - qzz

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
    Elastic scattering needs to satisfy k**2 = qx**2 + qy**2 + qz**2
    Given a qx and qy, find the correct qz value and interpolate the correct Escatter values

    input:
        qx - x-component of the scattering vector (NumX,)
        qy - y-component of the scattering vector (NumY,)
        qz - z-component of the scattering vector (NumZ,)
        escat_x, escat_y, escat_z - x, y, z components of the far-field projection (NumZ, NumY, NumX, 3)
        kmag - scattering vector magnitude (scalar)
    
    output:
        Iscatter - Scattering intensity, interpolated onto the Ewald Sphere (NumY, NumX)
    '''
    Iscatter = np.zeros((Pq.shape[1],Pq.shape[2]))
    for i in range(len(qy)):
        for j in range(len(qx)):
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
