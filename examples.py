''' This file is to hold some basic functions that may be useful for running some simple examples'''
import math as m
import numpy as np
import gtsam

def modulo_idx(vector,idx):
    '''
    This function takes in a numpy array of a short length and an index
    and returns a control such that the (first) array index repeats.
    e.g. modulo_idx(v,2) and modulo_idx(v,5) of a 3-element vector 'v' will
        return the same thing

    Args:
        vector: a numpy array to grab items out of
        idx: the idx that will be "folded" back into the length of vector

    Returns:
        An element from 'vector'
    '''
    return vector[idx % len(vector)]

def dynamics(curr_pose,V,w,dt,bias=0., add_noise=None):
    '''
    Takes in and returns a gtsam.Pose2 object

    Args: 
        curr_pose: A gtsam Pose2 object
        V: the velocity (in m/s)
        w (omega): the angular rate (in radians/s)
        dt: the delta time to apply the current V and w at (in seconds)
        bias: If given, will offset w by the bias
        add_noise: If given, will add the values in add_noise (a 3-element thingy) to the output
    
    Returns:  a gtsam.Pose2 object
    '''
    #WARNING, any changes here should be reflected in dDynamics_... functions below
    tw = w - bias # true omega
    
    est_angle = curr_pose.theta() + (tw*dt)/2.
    next_x = curr_pose.x() + V*m.cos(est_angle)*dt
    next_y = curr_pose.y() + V*m.sin(est_angle)*dt
    next_theta = curr_pose.theta() + tw*dt
    if add_noise is not None:
        next_x += add_noise[0]
        next_y += add_noise[1]
        next_theta += add_noise[2]
    return gtsam.Pose2(next_x, next_y, next_theta)

def dDynamics_dpose(curr_pose,V,w,dt,bias=0.):
    '''
    Returns a 3x3 matrix with the derivative of the next pose w.r.t. the curr_pose

    Args: 
        curr_pose: A gtsam Pose2 object
        V: the velocity (in m/s)
        w (omega): the angular rate (in radians/s)
        dt: the delta time to apply the current V and w at (in seconds)
        bias: If given, will offset w by the bias

    Returns 3x3 numpy array
    '''
    #Tightly coupled to dynamics function above!
    tw = w - bias # true omega

    est_angle = curr_pose.theta() + (tw*dt)/2.

    going_out = np.eye(3)
    going_out[0,2] = -V*m.sin(est_angle)*dt
    going_out[1,2] = V*m.cos(est_angle)*dt

    return going_out

def pose2_list_to_nparray(pose2_list):
    going_out = np.zeros((len(pose2_list),3))
    for ii,cp in enumerate(pose2_list):
        going_out[ii] = np.array([cp.x(),cp.y(),cp.theta()])
    return going_out
