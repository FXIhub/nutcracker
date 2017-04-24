import h5py
import numpy as np
import nutcracker


def test_fourier_shell_correlation():
    with h5py.File('/scratch/fhgfs/doctor/1FFK/models/1FFK_model_hc_fs.h5', "r") as f:
        img = f['real']

    fsc = nutcracker.fourier_shell_correlation(img,img)

    if fsc.mean() == 1.0:
        print 'passed'
    else:
        print 'failed'
