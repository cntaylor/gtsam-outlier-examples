import gtsam
import numpy as np
import math as m
from typing import List, Optional

def pose2_list_to_nparray(pose2_list: List[gtsam.Pose2]) -> np.ndarray:
    '''
    Converts a list of gtsam Pose2 types into a numpy array with shape Nx3,
    where the three columns correspond to x, y, and theta.
    '''
    going_out = np.zeros((len(pose2_list), 3))
    for ii, cp in enumerate(pose2_list):
        going_out[ii] = np.array([cp.x(), cp.y(), cp.theta()])
    return going_out

def angleize_np_array(np_array: np.ndarray) -> np.ndarray:
    ''' 
    Take in a numpy array and ensure all values are valid angles (between -pi and pi)
    by adding and subtracting 2*pi values to them
    '''
    while np.any(np_array < -m.pi):
        np_array[np_array < -m.pi] += 2 * m.pi
    while np.any(np_array > m.pi):    
        np_array[np_array > m.pi] -= 2 * m.pi
    return np_array

def error_range_known_landmark(landmark_loc: np.ndarray, measurement: float, 
                               this: gtsam.CustomFactor, values: gtsam.Values, 
                               jacobians: Optional[List[np.ndarray]]) -> np.ndarray:
    '''
    This is a custom factor for GTSAM. It takes in a known landmark location and the 
    measured range, returning the error and the Jacobians of the measurement (as needed)
    
    Inputs:
        landmark_loc: a 2-element numpy vector that has the location of the landmark
        measurement: scalar measurement between current robot location and landmark
        this: Makes it callable by gtsam as a custom factor
        values: gtsam.Values, but in this case should give me a robot location (Pose2)
        jacobians: If required, lets me pass out the H matrix (d measurement / d pose2)
    
    Output: the error between the predicted range and the measurements (h(x) - z) 
    '''

    key = this.keys()[0]
    est_loc = values.atPose2(key)
    np_est_loc = np.array([est_loc.x(), est_loc.y(), est_loc.theta()])
    diff_loc = np_est_loc[:2] - landmark_loc
    pred_range = np.sqrt(np.sum(np.square(diff_loc)))
    error = pred_range - measurement 
    # print('Error',error,'pred_range',pred_range,'measurement',measurement,'landmakr_loc',landmark_loc)

    if jacobians is not None:
        # Have to be careful with Jacobians. They are not with respect to the
        # full state, but rather the error state.  
        range_deriv = np.array([diff_loc[0]/pred_range, diff_loc[1]/pred_range, 0])
        # Now rotate into the error space for Pose2
        theta = est_loc.theta()
        DCM = np.array([np.cos(theta), -np.sin(theta), np.sin(theta), np.cos(theta)]).reshape(2,2)
        range_deriv[:2] = range_deriv[:2] @ DCM
        jacobians[0] = range_deriv.reshape(1,3)
    
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

def switchable_error_range_known_landmark(landmark_loc: np.ndarray, measurement: float, 
                               this: gtsam.CustomFactor, values: gtsam.Values, 
                               jacobians: Optional[List[np.ndarray]]) -> np.ndarray:
    '''
    This is a custom factor for GTSAM. It takes in a known landmark location and the 
    measured range, returning the error and the Jacobians of the measurement (as needed).
    Assume you will pass in the pose and the switch hidden variables (in that order)
    
    Inputs:
        landmark_loc: a 2-element numpy vector that has the location of the landmark
        measurement: scalar measurement between current robot location and landmark
        this: Makes it callable by gtsam as a custom factor
        values: gtsam.Values, but in this case should give me a robot location (Pose2) 
                and a switch (float) value
        jacobians: If required, lets me pass out the H matrix (d (measurement*switch) / d pose2) &
                    (d (measurement*switch) / d switch)
    
    Output: the error between the predicted range and the measurements (h(x) - z), 
            weighted by the scaling factor
    '''

    # Wanted to make this a parameter that can be passed in, but turns out
    # to be rather difficult with how this interacts with GTSAM.  So,
    # it is now set at the beginning of the function :(
    epsilon = 1E-5
    
    key_pose = this.keys()[0]
    est_loc = values.atPose2(key_pose)
    # I don't know if this is needed or not, but if the switch goes negative, that would
    # throw off the math, so just make sure it doesnt... (while getting the value)
    sw_key = this.keys()[1] # switching constraint key
    orig_switch = max(0,values.atDouble(sw_key))
    # I do sqrt because I want the switch to scale the squared error, not (possibly negative) error
    switch = m.sqrt(orig_switch)
    np_est_loc = np.array([est_loc.x(), est_loc.y(), est_loc.theta()])
    diff_loc = np_est_loc[:2] - landmark_loc
    pred_range = np.sqrt(np.sum(np.square(diff_loc)))
    uw_error = pred_range - measurement # unweighted error
    error = uw_error * switch
    # print('Error',error,'pred_range',pred_range,'measurement',measurement,'landmakr_loc',landmark_loc)

    if jacobians is not None:
        # Have to be careful with Jacobians. They are not with respect to the
        # full state, but rather the error state.  
        range_deriv = np.array([diff_loc[0]/pred_range, diff_loc[1]/pred_range, 0])
        # Now rotate into the error space for Pose2
        theta = est_loc.theta()
        DCM = np.array([np.cos(theta), -np.sin(theta), np.sin(theta), np.cos(theta)]).reshape(2,2)
        range_deriv[:2] = switch * range_deriv[:2] @ DCM
        jacobians[0] = range_deriv.reshape(1,3)
        # Maybe I don't need to, but I am worried about the / orig_switch value (the correct value)
        # being numerically stable as the switch value approaches 0
        jacobians[1] = np.array([uw_error * 0.5/(switch+epsilon)]).reshape(1,1)
    
    return np.array([error])

