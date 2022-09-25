'''
Collection of functions to rotate optical tensors
'''

import numba as nb
import warnings
from cuda import einsum_gpu, rotate_n_gpu

### Try importing cupy to replace numpy. Fall back to numpy if cupy isn't installed
try:
    import cupy as np
    MACHINE_HAS_CUDA = True
except ModuleNotFoundError:
    import numpy as np
    warnings.warn('No cupy installation found. Falling back to numpy and numba for calculations')
    MACHINE_HAS_CUDA = False



def einsum(array1, array2,backend='cupy', TPB_x=32):
    '''
    Wrapper function to perform optical tensor rotation. 
    Will execute on GPU with cupy or Numba CUDA kernel, or CPU with Numba-compiled function

    Parameters
    ----------
        array1 : ndarray
            n-dimensional array, where the size of the last two dimensions is 3
        array2 : ndarray
            n-dimensional array, where the size of the last two dimensions is 3
        backend : str
            Specify whether to use cupy or numba cuda backend on GPU
        TPB_x : int
            Number of threads per block to use
    
    Returns
    -------
        array_out : ndarray
            n-dimensional array of the same shape as array1 & array 2
    '''
    if MACHINE_HAS_CUDA:
        if backend == 'cupy':
            return np.einsum('aij,ajk->aik', array1, array2)
        else:
            return einsum_gpu(array1, array2, TPB_x=TPB_x)
    else:
        return einsum_cpu(array1, array2)


def rotate_n(n, rotmat, backend='cupy', TPB_x=32):
    '''
    Wrapper function to perform optical tensor rotation. 
    Will execute on GPU with cupy or Numba CUDA kernel, or CPU with Numba-compiled function

    Parameters
    ----------
        n : 3x3 array, complex
            Complex optical tensor of the form n = 1 - delta + i*beta
        rotmat : ndarray
            Rotation matrices for each voxel in the morphology [NumZ, NumY, NumX, 3, 3]
        backend : str
            Specify whether to use cupy or numba cuda backend on GPU
        TPB_x : int
            Number of threads per block to use
    
    Returns
    -------
        n_rotated : ndarray, complex
            Rotated complex optical tensor for each voxel in the morphology [NumZ, NumY, NumX, 3, 3]
    '''
    if MACHINE_HAS_CUDA:
        if backend == 'cupy':
            return np.einsum('aij,ajk->aik', rotmat, np.einsum('ij,akj->aik',n,rotmat))
        else:
            return rotate_n_gpu(n, rotmat, TPB_x=TPB_x)
    else:
        return rotate_n_cpu(n, rotmat)


def create_Rzyz(phi, theta, psi):
    '''
    Creates the full set of 3x3 ZYZ rotation matrices from Euler Angles
    
    Parameters
    ----------
        phi : ndarray
            1st rotation angle, about the Z-axis [NumZ, NumY, NumX]
        theta : ndarray
            2nd rotation angle, about the Y-axis [NumZ, NumY, NumX]
        psi : ndarray
            3rd rotation angle, about the Z-axis [NumZ, NumY, NumX]
    
    Returns
    -------
        r_zyz : ndarray
            ZYZ rotation matrices [NumZ, NumY, NumX, 3, 3]
    '''

    rz1 = create_Rz(phi)
    ry = create_Ry(theta)
    rz2 = create_Rz(psi)
    r_zyz = einsum(rz2,einsum(ry,rz1))
    return r_zyz

def create_Rz(angles):
    '''
    Creates rotation matrices about Z for each voxel in the morphology. 
    Uses numpy/cupy broadcasting and is faster without numba compilation

    Parameters
    ----------
        angles : ndarray
            Array of angles (in radians) for rotation about the Z-axis. Shape [NumZ, NumY, NumX]
    
    Returns
    -------
        rz : ndarray
            Array of 3x3 rotation matrices. Shape [NumZ, NumY, NumX, 3, 3]
    '''
    rz = np.zeros((*angles.shape,3,3))
    tmp_cos = np.cos(angles)
    tmp_sin = np.sin(angles)
    rz[...,0,0] = tmp_cos
    rz[...,1,1] = tmp_cos
    rz[...,1,0] = tmp_sin
    rz[...,0,1] = -tmp_sin
    rz[...,2,2] = 1
    return rz

def create_Ry(angles):
    '''
    Creates rotation matrices about Y for each voxel in the morphology. 
    Uses numpy/cupy broadcasting and is faster without numba compilation

    Parameters
    ----------
        angles : ndarray
            Array of angles (in radians) for rotation about the Y-axis. Shape [NumZ, NumY, NumX]
    
    Returns
    -------
        ry : ndarray
            Array of 3x3 rotation matrices. Shape [NumZ, NumY, NumX, 3, 3]
    '''
    ry = np.zeros((*angles.shape,3,3))
    tmp_cos = np.cos(angles)
    tmp_sin = np.sin(angles)
    ry[...,0,0] = tmp_cos
    ry[...,2,2] = tmp_cos
    ry[...,2,0] = -tmp_sin
    ry[...,0,2] = tmp_sin
    ry[...,1,1] = 1
    return ry

def create_Rx(angles):
    '''
    Creates rotation matrices about X for each voxel in the morphology. 
    Uses numpy/cupy broadcasting and is faster without numba compilation

    Parameters
    ----------
        angles : ndarray
            Array of angles (in radians) for rotation about the X-axis. Shape [NumZ, NumY, NumX]
    
    Returns
    -------
        rx : ndarray
            Array of 3x3 rotation matrices. Shape [NumZ, NumY, NumX, 3, 3]
    '''
    rx = np.zeros((*angles.shape,3,3))
    tmp_cos = np.cos(angles)
    tmp_sin = np.sin(angles)
    rx[...,0,0] = 1
    rx[...,1,1] = tmp_cos
    rx[...,2,2] = tmp_cos
    rx[...,2,1] = tmp_sin
    rx[...,1,2] = -tmp_sin
    return rx

@nb.njit(parallel=True)
def einsum_cpu(array1, array2):
    '''
    numba compiled and optimized function that replaces 
    np.einsum for the operation '...ij,...jk->...ik'

    Parameters
    ----------
        array1 : ndarray
            [NumZ, NumY, NumX, 3, 3]
        array2 : ndarray
            [NumZ, NumY, NumX, 3, 3]
    
    Returns
    -------
        result: ndarray
            [NumZ, NumY, NumX, 3, 3]
    '''

    shape = array1.shape
    array1 = np.reshape(array1,(-1,3,3))
    array2 = np.reshape(array2,(-1,3,3))
    result = np.empty(array1.shape)

    for s in nb.prange(array1.shape[0]):
        for i in range(3):
            for k in range(3):
                acc = 0
                for j in range(3):
                    acc += array1[s,i,j]*array2[s,j,k]
                result[s,i,k] = acc

    result = np.reshape(result,shape)
    return result

@nb.njit(parallel=True)
def rotate_n_cpu(n,rotmat):
    '''
    Numba implementation of rotating [3,3] optical tensor across morphology array.
    Performs successive contractions of 'ij,akj->aik' and 'aij,ajk->aik'. 
    Equivalent to R@n@R.T or np.einsum, but faster

    Parameters
    ----------

        n : ndarray
            Aligned optical tensor (biaxial or uniaxial) [3,3]
        rotmat : ndarray
            Rotation matrices [NumZ, NumY, NumX, 3, 3]

    Returns
    -------
        n_rotated : ndarray, complex
        Rotated optical tensors (NumZ, NumY, NumX, 3, 3)
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



