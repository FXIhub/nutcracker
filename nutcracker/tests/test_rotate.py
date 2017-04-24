import h5py
import numpy as np
import nutcracker

def test_rotation_matrix():
    axis = ['x','y','z']
    for i in axis:
        rot_mat = nutcracker.utils.rotation_matrix(0,axis[i])
        if rot_mat.sum() != 3.0:
            print 'failed, axis: %s' %(axis[i])
        else:
            print 'passed'
