import h5py
import pathlib
import numba as nb



### Try importing cupy to replace numpy. Fall back to numpy if cupy isn't installed
try:
    import cupy as np
except ModuleNotFoundError:
    import numpy as np



class Morphology():

    def __init__(self):
        