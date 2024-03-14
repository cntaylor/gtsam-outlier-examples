import gtsam
import numpy as np
import numpy.linalg as la
import math as m
from typing import List, Optional
import numpy as np

time_divider=1E6
c = 2.99792458E8 # m/s, speed of light
c_small = c/time_divider # speed of light in smaller time units (to make solving more numerically stable)

def get_chemnitz_data(filename = 'Data_Chemnitz.csv') -> np.array:
    '''
    Read the chemnitz data. This code is full of magic numbers, etc.  The file I know of
    has 90273 rows of GNSS data, then about 8000 lines later starts on ground truth data.

    Returns:
        Big array of # times x 3.  Columns will be:
            0: Time in experiment
            1: ground truth (3 element numpy array -- x,y,z of GNSS receiver)
            2: GNSS data -- a (variable length) list of 5-element numpy arrays.  5 elements are 
                [satellite ID, pseudo-range, x,y,z of satellite]
    '''
    # TODO:  have it return a pandas table?

    gnss_data = np.genfromtxt(filename, delimiter=',', usecols=(1, 2, 4, 5, 6, 7), max_rows=90273, dtype=np.float64)
    #Start on line 98844 (at least in Excel :)
    gt_data = np.genfromtxt(filename, delimiter=',', usecols=(1, 2, 3, 4), skip_header=98843, dtype=np.float64)
    # Do some quick checks on the data
    assert np.all(np.diff(gnss_data[:,0]) >= 0), "GNSS times are not in order"
    assert np.all(np.diff(gt_data[:,0]) >= 0), "GT times are not in order"
    unique_gnss_times = np.unique(gnss_data[:,0])
    unique_gt_times = np.unique(gt_data[:,0])
    assert np.all(np.equal(unique_gnss_times, unique_gt_times)), "Times not the same between GNSS and Ground Truth!"
    going_out = np.empty((len(unique_gnss_times), 3), dtype=object)
    going_out[:,0] = unique_gnss_times
    gnss_idx = 0
    for i in range(len(gt_data)):
        going_out[i,1] = gt_data[i,1:4]
        list_data = []
        while gnss_idx < len(gnss_data) and gnss_data[gnss_idx,0] == unique_gnss_times[i]:
            # Flip the satellite ID to the front
            tmp_array = np.zeros(5)
            tmp_array[0] = gnss_data[gnss_idx,5]
            tmp_array[1:] = gnss_data[gnss_idx,1:5]
            list_data.append( tmp_array )
            gnss_idx += 1
        going_out[i,2] = list_data
    
    return going_out

def init_pos(pseudorange_list: List[np.ndarray],
             res_cov : Optional[np.array] = None) -> np.ndarray:
    '''
    This takes in a list of pseudoranges (the list put into the 
    3rd column in get_chemnitz_data) and returns the initial
    GNSS position and time (x,y,z,t,0)

    if res_cov is not None, return the estimated covariance of the position & time measurement in res_cov
    '''
    assert len(pseudorange_list) >= 4, "Need at least 4 pseudoranges"
    tmp_var = np.zeros(5)
    z = np.array([pseudorange_list[i][1] for i in range(len(pseudorange_list))])
    done = False
    # Run the G-N iteration
    while not done:
        # Matrix to perform G-N iterations with
        A = np.zeros((len(pseudorange_list), 4))
        est_pseudo = np.zeros(len(pseudorange_list))
        for i in range(len(pseudorange_list)):
            diff_loc = tmp_var[:3] - pseudorange_list[i][2:5]
            dist = np.sqrt(np.sum(np.square(diff_loc)))
            A[i,:3] = diff_loc/dist
            A[i,3] = c_small
            est_pseudo[i] = dist + tmp_var[3]*c_small
        y = z - est_pseudo
        delta_tmp_var = la.pinv(A) @ y
        tmp_var[:4] += delta_tmp_var
        if la.norm(delta_tmp_var) < 5.: # Don't need it really exact
            done=True
    if res_cov is not None:
        res_cov[:4,:4] = la.inv(A.T @ A) # Make it modify res_cov

    return tmp_var

def error_clock(time_diff: float, this: gtsam.CustomFactor, values: gtsam.Values, jacobians: Optional[List[np.ndarray]]) -> np.ndarray:
    '''
    This is a custom factor for GTSAM. It takes in the clock error
    and the clock drift, returning the error and the Jacobians of the measurement (as needed)
    
    Inputs:
        values: gtsam.Values, but in this case should give me a GNSS receiver state
                (5 element vector -- x,y,z, clock error, clock drift)
        jacobians: If required, lets me pass out the H matrix (d measurement / d pose2)
    
    Output: the error between the predicted range and the measurements (h(x) - z) 
    '''
    key1 = this.keys()[0]
    key2 = this.keys()[1]
    state1 = values.atVector(key1)
    state2 = values.atVector(key2)
    going_out = np.zeros(2)
    going_out[0] = state2[3] - state1[3] - state1[4]*time_diff
    going_out[1] = state2[4] - state1[4]
    # Now do the jacobians of the above formulas
    if jacobians is not None:
        J1 = np.zeros((2,5))
        J1[0,3] = -1.
        J1[0,4] = -time_diff
        J1[1,4] = -1.
        J2 = np.zeros((2,5))
        J2[0,3] = 1.
        J2[1,4] = 1.
        jacobians[0] = J1
        jacobians[1] = J2
    return going_out

def error_psuedorange(measurement: float, satellite_loc: np.ndarray, 
                      this: gtsam.CustomFactor, values: gtsam.Values, 
                      jacobians: Optional[List[np.ndarray]]) -> np.ndarray:
    '''
    This is a custom factor for GTSAM. It takes in a known satellite location and the 
    measured range, returning the error and the Jacobians of the measurement (as needed)
    
    Inputs:
        values: gtsam.Values, but in this case should give me a GNSS receiver state
                (5 element vector -- x,y,z, clock error, clock drift)
        jacobians: If required, lets me pass out the H matrix (d measurement / d pose2)
    
    Output: the error between the predicted range and the measurements (h(x) - z) 
    '''

    key = this.keys()[0]
    curr_state = values.atVector(key)
    diff_loc = curr_state[:3] - satellite_loc
    pred_range = np.sqrt(np.sum(np.square(diff_loc))) 
    error = pred_range + curr_state[3]*c_small - measurement 
    # print('Error',error,'pred_range',pred_range,'measurement',measurement,'landmakr_loc',landmark_loc)

    if jacobians is not None:
        prange_deriv = np.zeros((1,5))
        prange_deriv[0,:3] = diff_loc/pred_range
        prange_deriv[0,3] = c_small
        jacobians[0] = prange_deriv

    
    return np.array([error])

def switchable_constraint_error(this: gtsam.CustomFactor, values: gtsam.Values,
                                jacobians: Optional[List[np.ndarray]]) -> np.ndarray:
    '''
    This is just me testing out the "Prior" factor as it does things
    I'm not expecting.  This tests if it does what I think it should
    '''
    switch_value = values.atDouble(this.keys()[0])
    if jacobians is not None:
        jacobians[0] = np.array([1])
    return np.array([switch_value-1.0])

def switchable_error_pseudorange(satellite_loc: np.ndarray, measurement: float,
                                 this: gtsam.CustomFactor, values: gtsam.Values, 
                                 jacobians: Optional[List[np.ndarray]]) -> np.ndarray:
    '''
    This is a custom factor for GTSAM. It takes in a known satellite location and the 
    measured pseudo range, returning the error and the Jacobians of the measurement (as needed).
    Assume you will pass in the state and the switch hidden variables (in that order)
    
    Inputs:
        satellite_loc: a 3-element vector (x,y,z)
        measurement: pseudo-range measurement between receiver location and satellite
        this: Makes it callable by gtsam as a custom factor
        values: gtsam.Values, but in this case should give me a state (5-element vector)
                and a switch (float) value
        jacobians: If required, lets me pass out the H matrix (d (measurement*switch) / d state) &
                    (d (measurement*switch) / d switch)
    
    Output: the error between the predicted range and the measurements (h(x) - z), 
            weighted by the scaling factor
    '''

    # Wanted to make this a parameter that can be passed in, but turns out
    # to be rather difficult with how this interacts with GTSAM.  So,
    # it is now set at the beginning of the function :(
    epsilon = 1E-5
    
    key_pose = this.keys()[0]
    curr_state = values.atVector(key_pose)
    # print('Error',error,'pred_range',pred_range,'measurement',measurement,'landmakr_loc',landmark_loc)

    # I don't know if this is needed or not, but if the switch goes negative, that would
    # throw off the math, so just make sure it doesnt... (while getting the value)
    sw_key = this.keys()[1] # switching constraint key
    orig_switch = max(0,values.atDouble(sw_key))
    # I do sqrt because I want the switch to scale the squared error, not (possibly negative) error
    switch = m.sqrt(orig_switch)
    diff_loc = curr_state[:3] - satellite_loc
    pred_range = np.sqrt(np.sum(np.square(diff_loc))) 
    uw_error = pred_range + curr_state[3]*c_small - measurement 
    error = uw_error * switch

    if jacobians is not None:
        prange_deriv = np.zeros((1,5))
        prange_deriv[0,:3] = diff_loc/pred_range
        prange_deriv[0,3] = c_small
        jacobians[0] = prange_deriv * switch

        # Maybe I don't need to, but I am worried about the / orig_switch value (the correct value)
        # being numerically stable as the switch value approaches 0
        jacobians[1] = np.array([uw_error * 0.5/(switch+epsilon)]).reshape(1,1)
    
    return np.array([error])

if __name__ == "__main__":
    gnss_data = get_chemnitz_data()
    print(gnss_data.shape)
    print('# of GNSS times is',len(gnss_data))
