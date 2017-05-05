import numpy as np
import condor

def compare_two_sets_of_quaternions(q1,q2,full_output=False,n_samples=100,q1_is_extrinsic=False,q2_is_extrinsic=False,sigma=2):
    """
    Compares two sets of quaternions based on a comparison of randomly picked quaternions within each set. 

    Args: 
        :q1(float ndarray):        5d ndarray of intrinsic quaternions
        :q2(float ndarray):        5d ndarray of intrinsic quaternions

    Kwargs:
        :full_output(bool):        returns full output as a dictionary, default = False
        :n_samples(int):           number of randomly picked pairs of quaternions, default = 100
        :q1_is_extrinsic(bool):    q1 quaternions are extrinsic and should be converted to instrinsic, default = False
        :q2_is_extrinsic(bool):    q2 quaternions are extrinsic and should be converted to instrinsic, default = False 
        :sigma(int):               sigma of standart deviation, default = 2
    """

    q1_rel_list = []
    q2_rel_list = []
    theta_rel = []

    # iterate through a number of randomly picked pairs
    for i in range(n_samples):
        
        # pick random index
        m = np.random.randint(0,q1.shape[0])
        n = np.random.randint(0,q1.shape[0])
        
        # make sure m and n are not the same
        if m == n: n = q1.shape[0] - n
        
        # pick random quaternions and make sure they are represented on one half of the hyper sphere
        q1_m = q1[m,:]
        q1_n = q1[n,:]
        q2_m = q2[m,:]
        q2_n = q2[n,:]

        # normalise quaternions
        q1_m = q1_m/np.sqrt(q1_m[0]**2 + q1_m[1]**2 + q1_m[2]**2 + q1_m[3]**2)
        q1_n = q1_n/np.sqrt(q1_n[0]**2 + q1_n[1]**2 + q1_n[2]**2 + q1_n[3]**2)
        q2_m = q2_m/np.sqrt(q2_m[0]**2 + q2_m[1]**2 + q2_m[2]**2 + q2_m[3]**2)
        q2_n = q2_n/np.sqrt(q2_n[0]**2 + q2_n[1]**2 + q2_n[2]**2 + q2_n[3]**2)
    
        # convert quaternions if necessary
        if q1_is_extrinsic: 
            q1_m = condor.utils.rotation.quat_conj(q1_m)
            q1_n = condor.utils.rotation.quat_conj(q1_n)
        if q2_is_extrinsic: 
            q2_m = condor.utils.rotation.quat_conj(q2_m)
            q2_n = condor.utils.rotation.quat_conj(q2_n)

        # calculating the inverse of q -> q^-1
        q1_n_inv = condor.utils.rotation.quat_conj(q1_n)/(np.sqrt(q1_n[0]**2 + q1_n[1]**2 + q1_n[2]**2 + q1_n[3]**2)**2)
        q2_n_inv = condor.utils.rotation.quat_conj(q2_n)/(np.sqrt(q2_n[0]**2 + q2_n[1]**2 + q2_n[2]**2 + q2_n[3]**2)**2)
    
        # calculating the relative quaternion q_rel = q_m * q_n^-1
        q1_rel = condor.utils.rotation.quat_mult(q1_n_inv, q1_m)
        w1 = q1_rel[0]

        # calculating the relative quaternion q_rel = q_m * q_n^-1 
        q2_rel_1 = condor.utils.rotation.quat_mult(q2_n_inv, q2_m)
        q2_rel_2 = condor.utils.rotation.quat_mult(q2_n_inv, -1 * q2_m)
        w2_1 = q2_rel_1[0]
        w2_2 = q2_rel_2[0]
        if np.abs(w1 - w2_1) < np.abs(w1 - w2_2): 
            w2 = w2_1
            q2_rel = q2_rel_1
        else:
            w2 = w2_2
            q2_rel = q2_rel_2
        
        # calculating the angle between the relative rotations
        q_rel = condor.utils.rotation.quat_mult(q1_rel,condor.utils.rotation.quat_conj(q2_rel))
        theta_rel.append(2 * np.arctan2(np.sqrt(q_rel[1]**2 + q_rel[2]**2 + q_rel[3]**2),np.abs(q_rel[0])))
        #theta_rel.append(2 * np.arccos(np.inner(q1_rel,q2_rel)))

        q1_rel_list.append(q1_rel)
        q2_rel_list.append(q2_rel)
    
    q1_rel_array = np.array(q1_rel_list)
    q2_rel_array = np.array(q2_rel_list)
    theta_rel = np.array(theta_rel)
        
    # difference between scalar components of two relative quaternions
    diff_rel = q1_rel_array[:,0] - q2_rel_array[:,0]
    diff_rel_mean = diff_rel.mean()
    diff_rel_std = diff_rel.std()

    # calculate z-score
    z_score = (diff_rel - diff_rel_mean)/diff_rel_std
    
    # percentage of quaternions within given sigma
    p = 1.0 * (z_score < sigma).sum()/n_samples

    if full_output:
        out = {'q1_rel':q1_rel_array,
               'q2_rel':q2_rel_array,
               'theta_rel':theta_rel,
               'diff_rel':diff_rel,
               'z_score':z_score}
        return p,out
    else:
        return p






def global_quaternion_rotation_between_two_sets(q1,q2,full_output=False,q1_is_extrinsic=False,q2_is_extrinsic=False,sigma=2):
    """
    Calculating the global rotation (quaternion) between two sets of quaternions on base of the mean relative quaternion between each sample of the sets.

    Args:
        :q1(float ndarray):        5d ndarray of intrinsic quaternions
        :q2(float ndarray):        5d ndarray of intrinsic quaternions

    Kwargs:
        :full_output(bool):        returns full output as a dictionary, default = False
        :q1_is_extrinsic(bool):    q1 quaternions are extrinsic and should be converted to instrinsic, default = False  
        :q2_is_extrinsic(bool):    q2 quaternions are extrinsic and should be converted to instrinsic, default = False
        :sigma(int):               sigma of standart deviation, default = 2
    """

    quat_list = []  
    
    for i in range(q1.shape[0]): 

        # iterate quaternions of the two sets
        q1_i = q1[i,:]
        q2_i = q2[i,:]

        #make sure they are represented on one half of the hyper sphere
        #if q1_i[0] < 0: q1_i[0] = -q1_i[0]
        #if q2_i[0] < 0: q2_i[0] = -q2_i[0]

        # normalising quaternions
        q1_i = q1_i/np.sqrt(q1_i[0]**2 + q1_i[1]**2 + q1_i[2]**2 + q1_i[3]**2)
        q2_i = q2_i/np.sqrt(q2_i[0]**2 + q2_i[1]**2 + q2_i[2]**2 + q2_i[3]**2)

        # convert quaternions if necessary
        if q1_is_extrinsic: q1_i = condor.utils.rotation.quat_conj(q1_i)
        if q2_is_extrinsic: q2_i = condor.utils.rotation.quat_conj(q2_i)
    
        # calculating the inverse quaternion of q1_i
        q1_inv = condor.utils.rotation.quat_conj(q1_i)/(np.sqrt(q1_i[0]**2 + q1_i[1]**2 + q1_i[2]**2 + q1_i[3]**2)**2)
    
        # calculating the relative quaternion between the two sets for each sample
        q_rel = condor.utils.rotation.quat_mult(q1_inv, q2_i)
    
        quat_list.append(q_rel)
    
    quat_array = np.array(quat_list)
    quat_array_mean = quat_array.sum(axis=0)/(1.0 * q1.shape[0])
    quat_array_mean = quat_array_mean/np.sqrt(quat_array_mean[0]**2 + quat_array_mean[1]**2 + quat_array_mean[2]**2 + quat_array_mean[3]**2)
    
    # calculate the angular uncertainty of the global relative quaternion
    angle_error = []

    for i in range(q1.shape[0]):

        # iterate through the sets of quaternions
        q1_i = condor.utils.rotation.unique_representation_quat(q1[i,:])
        q2_i = condor.utils.rotation.unique_representation_quat(q2[i,:])

        # normalising quaternions                                                                                                                                       
        q1_i = q1_i/np.sqrt(q1_i[0]**2 + q1_i[1]**2 + q1_i[2]**2 + q1_i[3]**2)
        q2_i = q2_i/np.sqrt(q2_i[0]**2 + q2_i[1]**2 + q2_i[2]**2 + q2_i[3]**2)

        # convert quaternions if necessary
        if q1_is_extrinsic: q1_i = condor.utils.rotation.quat_conj(q1_i)
        if q2_is_extrinsic: q2_i = condor.utils.rotation.quat_conj(q2_i)

        # calculate the relative rotation to q2
        q2_rot = condor.utils.rotation.quat_mult(q1_i,quat_array_mean)
        
        # calculating the angle between the actuall quaternion q1 and the roatation of q2 (q_rel = q1 * q2^-1  ->  q_rel * q2^-1 = q1)
        q2_rot_rel_1 = condor.utils.rotation.quat_mult(q2_i,condor.utils.rotation.quat_conj(q2_rot))
        q2_rot_rel_2 = condor.utils.rotation.quat_mult(-q2_i,condor.utils.rotation.quat_conj(q2_rot))

        theta_1 = 2 * np.arctan2(np.sqrt(q2_rot_rel_1[1]**2 + q2_rot_rel_1[2]**2 + q2_rot_rel_1[3]**2),np.abs(q2_rot_rel_1[0]))
        theta_2 = 2 * np.arctan2(np.sqrt(q2_rot_rel_2[1]**2 + q2_rot_rel_2[2]**2 + q2_rot_rel_2[3]**2),np.abs(q2_rot_rel_2[0]))

        #theta_1 = 2 * np.arccos(np.inner(q2_rot,q2_i))
        #theta_2 = 2 * np.arccos(np.inner(q2_rot,-1 * q2_i))

        # distinguish between the smaler angle
        if theta_1 < theta_2:
            angle_error.append(theta_1)
        else:
            angle_error.append(theta_2)
    
    angle_error = np.array(angle_error)
    angle_error_mean = angle_error.mean()
    angle_error_std = angle_error.std()

    # z-score
    z_score = (angle_error - angle_error_mean)/angle_error_std

    # percentage of quaternions within given sigma                                               
    p = 1.0 * (z_score < sigma).sum()/q1.shape[0]

    
    if full_output:
        out = {'quat_array':quat_array,
               'quat_array_mean':quat_array_mean,
               'angle_error':angle_error,
               'angle_error_mean':angle_error_mean,
               'angle_error_std':angle_error_std,
               'z_score':z_score,
               'percentage':p}
        return out
    else:
        return quat_array_mean, angle_error_mean
