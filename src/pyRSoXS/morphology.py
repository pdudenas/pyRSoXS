from distutils.errors import DistutilsModuleError
import h5py
import pathlib
import numba as nb
from rotation import MACHINE_HAS_CUDA
import CyRSoXS as cy


### Try importing cupy to replace numpy. Fall back to numpy if cupy isn't installed
if MACHINE_HAS_CUDA:
    import cupy as np
else:
    import numpy as np



class Morphology:

    def __init__(self, materials=dict(), PhysSize=None, ordering='ZYZ',NumZYX=None, NumMaterial=0, 
                config = {'CaseType':0, 'MorphologyType': 0, 'Energies': [270.0], 'EAngleRotation':[0.0, 1.0, 0.0]}):
        self.PhysSize = PhysSize
        self.ordering = ordering
        self.NumZYX = NumZYX
        self.NumMaterial = NumMaterial
        self.config = config
        self.materials = materials


    def __repr__():
        return 'Morphology (NumMaterial : {self.NumMaterial}, PhysSize : {self.PhysSize})'
    

    @classmethod
    def load_hdf5(cls,hdf5_file):
        with h5py.File(hdf5_file,'r') as f:
            if 'Euler_Angles' not in f.keys():
                raise KeyError('Only the Euler Angle convention is supported')
            
    @classmethod
    def load_config(cls, config_file):
        pass
    
    @classmethod
    def load_constants(cls, optical_contants):
        pass

    #TODO : function to write morphology to HDF5
    def write_hdf5(self,):
        pass
    
    #TODO : function to write a config.txt file from config dict
    def write_config(self,):
        pass
    
    #TODO : function to write constants to MaterialX.txt files
    def write_constants(self,):
        pass
    
    #TODO : submit to CyRSoXS or pyRSoXS backends
    def run(self,):
        pass

    #TODO : call checkH5 to validate morphology
    def check_materials(self,):
        pass




class Material:

    def __init__(self,materialID=0, Vfrac=None, S=None, phi=None, theta=None, NumZYX=None, name='Empty Material') -> None:
        self.materialID = materialID
        self.Vfrac = Vfrac
        self.S = S
        self.phi = phi
        self.theta = theta
        self.NumZYX = NumZYX
        self.name = name
        if self.NumZYX is None:
            try:
                self.NumZYX = Vfrac.shape
            except AttributeError:
                pass
    

    def __repr__(self):
        return f'Material ({self.name}, ID : {self.MaterialID}, Shape : {self.NumZYX})'

class OpticalConstants:

    def __init__(self, energies, deltaPara, betaPara, deltaPerp, betaPerp, name= 'unknown'):
        self.energies = energies