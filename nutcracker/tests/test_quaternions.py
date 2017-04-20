import h5py
import numpy as np
import nutcracker

def test_relative_angle():
    with h5py.File('/scratch/fhgfs/doctor/1FFK/data_cxi/data_hc.cxi', "r") as f:
        q1 = f['particles']['particle_00']['extrinsic_quaternion'][:]
    q2 = np.loadtxt('/scratch/fhgfs/doctor/1FFK/emc/emc_hc/sigma_1.5-3.0/run_0008/best_quaternion_0049.data', usecols=range(2000, 4))

    p, out = nutcracker.compare_two_sets_of_quaternions(q1,q2,n_samples=2000, full_output=True, sigma=3)
    if p >= 0.99:
        return 'passed'
    else:
        return 'failed'


def test_global_rotation():
    with h5py.File('/scratch/fhgfs/doctor/1FFK/data_cxi/data_hc.cxi', "r") as f:
        q1 = f['particles']['particle_00']['extrinsic_quaternion'][:]
    q2 = np.loadtxt('/scratch/fhgfs/doctor/1FFK/emc/emc_hc/sigma_1.5-3.0/run_0008/best_quaternion_0049.data', usecols=range(2000, 4))

    out = nutcracker.global_quaternion_rotation_between_two_sets(q1,q2,full_output=True,q1_is_extrinsic=True)
    quat_array = out['quat_array'][:]
    sigma = 2
    z_score = (quat_array[:,0] - quat_array[:,0].mean())/quat_array[:,0].std()
    p = 1.0 * (z_score < sigma).sum()/q1.shape[0]
    if p >= 0.99:
        return 'passed'
    else:
        return 'failed'
