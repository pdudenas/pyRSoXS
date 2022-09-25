'''
Collection of cuda kernels & functions
These are slower than cupy.einsum if cupy.einsum has sufficient GPU memory available and the cutensornet backend.
These are more memory conservative and faster in memory limited situations. 
See https://docs.cupy.dev/en/stable/user_guide/memory.html
'''


from numba import cuda, float32, complex64

try:
    import cupy as np
except ModuleNotFoundError:
    import numpy as np

def einsum_gpu(array1, array2, TPB_x = 32):
    '''
    Performs the matrix contraction '...ij,...jk->...ik'

    Parameters
    ----------
        array1 : ndarray
            n-dimensional array, where the last two dimension sizes are 3
        array2 : ndarray
            n-dimensional array where the last two dimensions sizes are 3
    
    Returns
    -------
        array_out : ndarray
            n-dimensional array of the same shape as array1 & array 2
    '''
    shape = array1.shape
    #reshape arrays to [NumZ*NumY*NumX, 3, 3]
    array1 = np.reshape(array1, (-1,3,3))
    array2 = np.reshape(array2, (-1,3,3))
    #allocate array to return
    array_out = cuda.device_array(array1.shape,dtype=np.float32)

    #define TPB and BPG for execution on GPU
    threadsperblock = (TPB_x, 3, 3)
    blockspergrid = (int(np.ceil(array1.shape[0]/TPB_x)), 1, 1)
    einsum_kernel[blockspergrid, threadsperblock](array1, array2, array_out)
    # reshape to [NumZ, NumY, NumX, 3, 3]
    array_out = np.reshape(array_out,shape)
    return array_out

def rotate_n_gpu(n,rotmat, TPB_x = 32):
    '''
    Rotates the complex optical tensor using a rotation matrix

    Parameters
    ----------
        n : 3x3 array, complex
            Complex optical tensor of the form n = 1 - delta + i*beta
        rotmat : ndarray
            Rotation matrices for each voxel in the morphology [NumZ, NumY, NumX, 3, 3]
    
    Returns
    -------
        n_rotated : ndarray, complex
            Rotated complex optical tensor for each voxel in the morphology [NumZ, NumY, NumX, 3, 3]
    '''
    shape = rotmat.shape
    #reshape arrays to [NumZ*NumY*NumX, 3, 3]
    rotmat = np.reshape(rotmat,(-1,3,3))
    #allocated intermediate and final arrays
    result1 = cuda.device_array(rotmat.shape,dtype=np.complex64)
    n_rotated = cuda.device_array(rotmat.shape,dtype=np.complex64)
    
    #define TPB and BPG
    threadsperblock = (TPB_x, 3, 3)
    blockspergrid = (int(np.ceil(rotmat.shape[0]/TPB_x)), 1, 1)
    rotate_n_kernel[blockspergrid, threadsperblock](n, rotmat, result1, n_rotated)
    #reshape to [NumZ, NumY, NumX, 3, 3]
    n_rotated = np.reshape(n_rotated,shape)
    return n_rotated



@cuda.jit
def einsum_kernel(A,B,C):
    '''
    Cuda kernel for matrix multiplication of size [N, 3, 3]. 
    Performs the matrix contraction 'aij,ajk->aik'.
         
    Parameters
    ----------
    
    A : cuda or cupy array
        Input array 1
    B : cuda or cupy array
        Input array 2
    C : cuda or cupy array
        Output array
    
    '''
    x, y, z = cuda.grid(3)

    ty = cuda.threadIdx.y
    tz = cuda.threadIdx.z
    
    stride, _, _ = cuda.gridsize(3)    # blocks per grid
    for pos in range(x,A.shape[0],stride):
        C[pos, ty, tz] = A[pos, ty, 0] * B[pos, 0, tz] + A[pos, ty, 1] * B[pos, 1, tz] + A[pos, ty, 2] * B[pos, 2, tz]

                
@cuda.jit
def rotate_n_kernel(n, rotmat, result1, n_rotated):
    '''
    Cuda kernel for rotating [3,3] optical tensor across morphology array.
    Performs successive contractions of 'ij,akj->aik' and 'aij,ajk->aik'. Equivalent to R@n@R.T.

    Parameters
    ----------
        n : ndarray
            Aligned optical tensor (biaxial or uniaxial) [3,3]
        rotmat : ndarray
            Reshaped rotation matrices [NumZ*NumY*NumX, 3, 3]
        result1 : ndarray
            Stores intermediate calculation result [NumZ*NumY*NumX, 3, 3]
        n_rotated : ndarray
            n-dimensional array of rotated optical tensors [NumZ*NumY*NumX, 3, 3]
    '''
    x, y, z = cuda.grid(3)

    ty = cuda.threadIdx.y
    tz = cuda.threadIdx.z

    stride, _, _ = cuda.gridsize(3)    # blocks per grid
    for pos in range(x,rotmat.shape[0],stride):
        result1[pos,ty,tz] = n[ty,0]*rotmat[pos,tz,0] + n[ty,1]*rotmat[pos,tz,1] + n[ty,2]*rotmat[pos,tz,2]
    
    cuda.syncthreads()
              
    for pos in range(x,rotmat.shape[0],stride):
        n_rotated[pos,ty,tz] = rotmat[pos,ty,0]*result1[pos,0,tz] + rotmat[pos,ty,1]*result1[pos,1,tz] + rotmat[pos,ty,2]*result1[pos,2,tz]