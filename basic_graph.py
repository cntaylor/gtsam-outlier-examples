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
import copy
from typing import List, Optional
from functools import partial
from tqdm import tqdm
import time
from unicycle_est_utils import pose2_list_to_nparray, error_range_known_landmark, angleize_np_array

DEBUG=False
if DEBUG:
    import matplotlib.pyplot as plt


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
    out_file = 'basic_graph_unicycle_res.npz'
    n_runs = 100
    # This is a data structure that holds the directory name and
    # what the output file should say so they get picked together!
    in_opts = np.array([
        ['No outliers', 'no_outliers/'],
        ['10% outliers', 'measurement_10pc_outliers/'],
        ['20% outliers', 'measurement_20pc_outliers/'],
        ['30% outliers', 'measurement_30pc_outliers/'],
        ['40% outliers', 'measurement_40pc_outliers/'],
        ['50% outliers', 'measurement_50pc_outliers/']
    ])
    # The basic Gaussian noise model assumed
    basic_meas_noise = \
        gtsam.noiseModel.Diagonal.Sigmas(np.diag(np.array([1.])))

    est_opts = np.array([
        ['no_outlier', basic_meas_noise],
        ['huber', gtsam.noiseModel.Robust.Create(
                            gtsam.noiseModel.mEstimator.Huber(k=1),
                            basic_meas_noise)],
        ['Cauchy', gtsam.noiseModel.Robust.Create(
                            gtsam.noiseModel.mEstimator.Cauchy(k=.1),
                            basic_meas_noise)],
        ['DCS-.5', gtsam.noiseModel.Robust.Create(
                            gtsam.noiseModel.mEstimator.DCS(c = 0.5),
                            basic_meas_noise)],
        ['DCS-1', gtsam.noiseModel.Robust.Create(
                            gtsam.noiseModel.mEstimator.DCS(c = 1.0),
                            basic_meas_noise)],
        ['DCS-2', gtsam.noiseModel.Robust.Create(
                            gtsam.noiseModel.mEstimator.DCS(c = 2.0),
                            basic_meas_noise)],
        ['Fair', gtsam.noiseModel.Robust.Create(
                            gtsam.noiseModel.mEstimator.Fair(1.3998),
                            basic_meas_noise)],
        ['Geman', gtsam.noiseModel.Robust.Create(
                            gtsam.noiseModel.mEstimator.GemanMcClure(1.0),
                            basic_meas_noise)],
        ['Tukey', gtsam.noiseModel.Robust.Create(
                            gtsam.noiseModel.mEstimator.Tukey(4.6851),
                            basic_meas_noise)],
        ['Welsch', gtsam.noiseModel.Robust.Create(
                            gtsam.noiseModel.mEstimator.Welsch(2.9846),
                            basic_meas_noise)]
    ])

    if DEBUG: # change this to know which one runs...
        in_opts = np.array([in_opts[0]])
        est_opts = np.array([est_opts[0]])
        which_run = 0
        run_list=[which_run]
        out_file = 'DEBUG'+out_file
    else:
        run_list = np.arange(n_runs)


    times = np.zeros((len(in_opts),len(est_opts),n_runs))
    pos_RMSEs = np.zeros((len(in_opts),len(est_opts),n_runs))
    ang_RMSEs = np.zeros((len(in_opts),len(est_opts),n_runs))

    # in_select and est_select control everything below
    for in_select in range(len(in_opts)):
        for est_select in range(len(est_opts)):
            print('Running input',in_opts[in_select,0], 'and estimator',est_opts[est_select,0])
            in_path = in_opts[in_select,1]
            # out_file = 'RMSE_input_'+in_opts[in_select,1]+'_est_'+est_opts[est_select,0]+'.npy'
            for i in tqdm(range(n_runs)):
                # First, read in the data from the file
                in_file = in_path+f'run_{i:04d}.npz'
                in_data = dict(np.load(in_file))

                in_data['x0'] = np.array([0, 0, m.pi/2])

                # Decide what cost function we will use for the measurements
                meas_noise = est_opts[est_select,1]

                ########   
                # Now run the optimziation (with whatever noise model you have)    
                start_time = time.time()
                initial_np, np_est_poses = solve_scenario(in_data, meas_noise=meas_noise)
                end_time = time.time()
                times[in_select,est_select,i] = end_time - start_time

                # plt.plot(np_est_poses)
                # plt.show()
                truth = in_data['truth']

                if DEBUG:
                    # When doing one run, good for plotting results
                    fig = plt.figure()

                    plt.plot(truth[:,0], truth[:,1])
                    plt.plot(np_est_poses[:,0], np_est_poses[:,1])
                    plt.plot(initial_np[:,0], initial_np[:,1])
                    plt.legend(['truth', 'est', 'initial'])
                    plt.show()

                RMSE = m.sqrt(np.average(np.square(truth[:,:2]- np_est_poses[:,:2])))
                RMSE_ang = m.sqrt(np.average( np.square( angleize_np_array(truth[:,2]- np_est_poses[:,2]) ) ) )
                if DEBUG:
                    print("RMSE (on x and y) is",RMSE)
                    print("RMSE (on angle) is",RMSE_ang)

                pos_RMSEs[in_select,est_select,i] = RMSE
                ang_RMSEs[in_select,est_select,i] = RMSE_ang
            # print("Average RMSEs (pos & angle) are",np.average(RMSEs,1))
            # plt.plot(RMSEs)
            # plt.show()
    # Required to load est_save back in from the .npz file
    est_save = est_opts[:,0].astype(str)
    in_opts_save = in_opts[:,0]
    np.savez(out_file, times=times, pos_RMSEs=pos_RMSEs, ang_RMSEs=ang_RMSEs, in_opts=in_opts_save, est_opts=est_save)
    print(est_save)


# %%
