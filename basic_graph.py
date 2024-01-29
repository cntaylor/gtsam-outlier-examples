#%%

'''
This creates and the estimates a simple unicycle robot 
(Pose2 type thing) with a ranging measurement to  
landmarks of known location and a simple dynamics model
where V (velocity) and w (turn rate) are the inputs
'''

import math as m
import numpy as np
import gtsam
import matplotlib.pyplot as plt
import copy
from typing import List, Optional
from functools import partial
from tqdm import tqdm

def pose2_list_to_nparray(pose2_list):
    '''
    A helper function that takes gtsam Pose2 types and converts into
    a numpy array that is Nx3, where the three columns correspond with
    x,y,theta
    '''
    going_out = np.zeros((len(pose2_list),3))
    for ii,cp in enumerate(pose2_list):
        going_out[ii] = np.array([cp.x(),cp.y(),cp.theta()])
    return going_out

def angleize_np_array(np_array):
    ''' 
    Take in an np array and make them all valid angles (between -pi and pi) by adding
    and subtracting 2*pi values to them
    '''
    while np.any(np_array < -m.pi):
        np_array[np_array < -m.pi] += 2 * m.pi
    while np.any(np_array > m.pi):    
        np_array[np_array > m.pi] -= 2 * m.pi
    return np_array

def error_range_known_landmark (landmark_loc : np.ndarray, measurement: float, 
                                this: gtsam.CustomFactor, values: gtsam.Values, 
                                jacobians: Optional[List[np.ndarray]]) -> np.ndarray:
    '''
    This is a custom factor for GTSAM.  It takes in a known landmark location and the 
    measured range, returning the error and the Jacobians of the measurement (as needed)
    
    Inputs:
        landmark_loc: a 2-element numpy vector that has the location of the landmark
        measurement:  scalar measurement between current robot location and landmark
        this:  Makes it callable by gtsam as a custom factor
        values:  gtsam stuff, but in this case should give me a robot location (Pose2)
        jacobians:  If required, lets me pass out the H matrix (d measurement / d pose2)
    
    Output:  the error between the predicted range and the measurements (h(x) - z) 
    '''

    key = this.keys()[0]
    est_loc = values.atPose2(key)
    np_est_loc = np.array([est_loc.x(), est_loc.y(), est_loc.theta()])
    diff_loc = np_est_loc[:2] - landmark_loc
    pred_range = m.sqrt( np.sum(np.square(diff_loc)) )
    error = pred_range - measurement 
    # print('Error',error,'pred_range',pred_range,'measurement',measurement,'landmakr_loc',landmark_loc)

    if jacobians is not None:
        # Have to be careful with Jacobians.  They are not with respect to the
        # full state, but rather the error state.  
        range_deriv = np.array([diff_loc[0]/pred_range, diff_loc[1]/pred_range, 0])
        # Now rotate into the error space for Pose2
        theta = est_loc.theta()
        DCM = np.array([m.cos(theta), -m.sin(theta),m.sin(theta), m.cos(theta)]).reshape(2,2)
        range_deriv[:2] = range_deriv[:2]@DCM
        jacobians[0] = range_deriv.reshape(1,3)
    
    return np.array([error])

def solve_scenario(in_data : dict, 
                   dt : float = .1,
                   meas_noise: gtsam.noiseModel = \
                    gtsam.noiseModel.Diagonal.Sigmas(np.diag(np.array([1.]))),
                   dyn_noise: gtsam.noiseModel = \
                    gtsam.noiseModel.Diagonal.Sigmas(np.array([.1,.1,.02]) * m.sqrt(.1))) \
                -> np.array:
    '''
    Take in a dictionary that has the 'measurements', 'inputs', 'x0', and 'landmarks' 
    locations in it.  Create a graph and return the optimized results as a np.array.  
    By passing in the noiseModels, allows the external user to try different Robust 
    (M-estimators) with the same function

    Inputs:
        input_meas_dict: the dictionary with the required information
        dt: the timestep between states (used with inputs to propagate)
        meas_noise:  What noise model to use with the measurements
        dyn_noise: What noise model to use with the dyanmics factors

    Outputs:
        A Nx3 np.array with the output poses

    '''
    Z = in_data['measurements'] # Z is the set of all measurements
    N = len(Z) - 1
    U = in_data['inputs'] # U is the set of all inputs, should be length N (which had -1 to get it)
    assert len(U)==N, "inputs and measurements have incompatible length"
    landmark_locs = in_data['landmarks']

    x0 = in_data['x0']

    # Create the graph and optimize
    graph = gtsam.NonlinearFactorGraph()
    initial_estimates = gtsam.Values()

    ## Functions for creating keys -- identifiers for hidden variables
    nl = len(landmark_locs) # number of landmarks
    pose_key = lambda x: gtsam.symbol('x',x)

    ## odometry factors
    ### Note that this uses Lie Algebra sorts of things. The factor is the difference between the 
    ### current and previous pose, in the the previous pose's coordinate frame.  So, it is a 
    ### differentiable Pose factor
    for ii in range(N):
        curr_V = U[ii,0] * dt
        curr_w = U[ii,1] * dt
        Vx = curr_V * m.cos(curr_w/2.)
        Vy = curr_V * m.sin(curr_w/2.)
        graph.add( gtsam.BetweenFactorPose2( pose_key(ii), pose_key(ii+1), gtsam.Pose2( Vx, Vy, curr_w ), dyn_noise ) )

    # measurement factors
    for ii,meas in enumerate(Z):
        for jj in range(nl):
            graph.add( gtsam.CustomFactor( meas_noise, [pose_key(ii)], 
                                            partial(error_range_known_landmark, landmark_locs[jj], meas[jj] ) ) )

    # Graph is formed, but need some initial values for the 
    # Use odometry only to initialize the graph.  Store the initial estimate as well for plotting (initial_np)
    initial_estimates.insert( pose_key(0), gtsam.Pose2(*x0) )
    curr_x=copy.copy( x0 )
    initial_np = np.zeros((N+1,3))
    initial_np[0] = x0
    for ii in range(N):
        curr_x[0] += U[ii,0]*dt * m.cos(curr_x[2]+U[ii,1]*dt/2.)
        curr_x[1] += U[ii,0]*dt * m.sin(curr_x[2]+U[ii,1]*dt/2.)
        curr_x[2] += U[ii,1]*dt
        initial_estimates.insert(pose_key(ii+1), gtsam.Pose2( *curr_x ) )
        initial_np[ii+1] = curr_x

    ## Everything should be set up. Now to optimize
    parameters = gtsam.GaussNewtonParams()
    parameters.setMaxIterations(100)
    parameters.setVerbosity("ERROR")
    optimizer = gtsam.GaussNewtonOptimizer(graph, initial_estimates, parameters)
    result = optimizer.optimize()

    # Prepare the results to be returned.
    est_poses=[]
    for ii in range(N):
        est_poses.append(result.atPose2(pose_key(ii)))
    est_poses.append(result.atPose2(pose_key(N)))
    np_est_poses = pose2_list_to_nparray(est_poses)

    return initial_np, np_est_poses

#%%
if __name__ == '__main__':
    n_runs = 100
    # This is a data structure that holds the directory name and
    # what the output file should say so they get picked together!
    in_opts = np.array([
        ['no_outliers/', 'no_outlier'],
        ['measurement_10pc_outliers/', 'meas_10pc'],
        ['measurement_20pc_outliers/', 'meas_20pc'],
        ['measurement_30pc_outliers/', 'meas_30pc'],
        ['measurement_40pc_outliers/', 'meas_40pc'],
        ['measurement_50pc_outliers/', 'meas_50pc'],
    ])
    est_opts = np.array([
        'no_outlier',
        'huber'
    ])
    # in_select and est_select control everything below
    for in_select in range(len(in_opts)):
        for est_select in range(len(est_opts)):
            in_select=0
            est_select=0

            in_path = in_opts[in_select,0]
            out_file = 'RMSE_input_'+in_opts[in_select,1]+'_est_'+est_opts[est_select]+'.npy'
            RMSEs = np.zeros((n_runs,2))
            for i in tqdm(range(n_runs)):
                # First, read in the data from the file
                in_file = in_path+f'run_{i:04d}.npz'
                in_data = dict(np.load(in_file))

                in_data['x0'] = np.array([0, 0, m.pi/2])

                # Decide what cost function we will use for the measurements
                # The basic Gaussian noise model assumed
                basic_meas_noise = \
                    gtsam.noiseModel.Diagonal.Sigmas(np.diag(np.array([1.])))
                
                if est_select == 0:
                    meas_noise=basic_meas_noise
                elif est_select == 1:
                    meas_noise = \
                        gtsam.noiseModel.Robust.Create(
                            gtsam.noiseModel.mEstimator.Huber(k=1),
                            basic_meas_noise)
                else:
                    print('Measurement noise model setup problem')
                ########   
                # Now run the optimziation (with whatever noise model you have)    
                initial_np, np_est_poses = solve_scenario(in_data, meas_noise= meas_noise)

                # plt.plot(np_est_poses)
                # plt.show()
                truth = in_data['truth']

                # # When doing one run, good for plotting results
                # fig = plt.figure()

                # plt.plot(truth[:,0], truth[:,1])
                # plt.plot(np_est_poses[:,0], np_est_poses[:,1])
                # plt.plot(initial_np[:,0], initial_np[:,1])
                # plt.legend(['truth', 'est', 'initial'])
                # plt.show()


                RMSE = m.sqrt(np.average(np.square(truth[:,:2]- np_est_poses[:,:2])))
                # print("RMSE (on x and y) is",RMSE)
                RMSE_ang = m.sqrt(np.average( np.square( angleize_np_array(truth[:,2]- np_est_poses[:,2]) ) ) )
                # print("RMSE (on angle) is",RMSE_ang)
                RMSEs[i] = np.array([RMSE,RMSE_ang])
            np.save(out_file,RMSEs)
            # print("Average RMSEs (pos & angle) are",np.average(RMSEs,1))
            # plt.plot(RMSEs)
            # plt.show()


# %%
