#%%

'''
This creates and the positions of a GNSS receiver from pseudo-ranges. 
For dynamics, it uses a simple "constant velocity" model and 
'''

import math as m
import numpy as np
import gtsam
import copy
from typing import List, Optional
from functools import partial
from tqdm import tqdm
import time
from gnss_est_utils import get_chemnitz_data, error_psuedorange, init_pos, error_clock


DEBUG=False
if DEBUG:
    import matplotlib.pyplot as plt


def solve_scenario(gnss_data : np.array, 
                   meas_noise: gtsam.noiseModel = \
                    gtsam.noiseModel.Diagonal.Sigmas(np.diag(np.array([1.]))),
                   dyn_Q_diags : np.array= np.array([4E-19, 1.4E-18]) ) \
                -> np.array:
    '''
    Take in a numpy array with all the data in it (see get_chemnitz_data for format),
    create a graph to handle it, and solve it
    Inputs:
        gnss_data: the array with all the measurements and timestamps in it
        meas_noise:  What noise model to use with the measurements
        dyn_noise: What noise model to use with the dyanmics factors

    Outputs:
        A Nx5 np.array with the output results

    '''

    # Create the graph and optimize
    graph = gtsam.NonlinearFactorGraph()
    initial_estimates = gtsam.Values()

    ## Functions for creating keys -- identifiers for hidden variables
    pose_key = lambda x: gtsam.symbol('x',x)

    ## odometry factors.  Basically, no odometry except on the clock...
    for ii in range(1,len(gnss_data)):
        dt = gnss_data[ii,0]-gnss_data[ii-1,0]
        graph.add( gtsam.CustomFactor( gtsam.noiseModel.Diagonal.Variances(dyn_Q_diags*dt), 
                                       [pose_key(ii-1), pose_key(ii)], 
                                       partial(error_clock, dt) ) )

    # measurement factors
    for ii, data in tqdm(enumerate(gnss_data)):
        meas_list = data[2]
        for jj in range(len(meas_list)):
            curr_meas=meas_list[jj]
            graph.add( gtsam.CustomFactor( meas_noise, [pose_key(ii)], 
                                            partial(error_psuedorange, curr_meas[1], curr_meas[2:]) ) )
        # Also add initial values
        initial_estimates.insert( pose_key(ii), init_pos( meas_list ) )

    ## Everything should be set up. Now to optimize
        # TODO:  Dogleg optimization
        # TODO: relativeDecrease termination of 1E-3
    parameters = gtsam.GaussNewtonParams()
    parameters.setMaxIterations(100)
    parameters.setVerbosity("ERROR")
    optimizer = gtsam.GaussNewtonOptimizer(graph, initial_estimates, parameters)
    result = optimizer.optimize()

    # Prepare the results to be returned.
    est_poses= np.zeros((len(gnss_data),5))
    for ii in range(len(gnss_data)):
        est_poses[ii] = result.atVector(pose_key(ii))

    return est_poses

#%%
if __name__ == '__main__':
    out_file = 'basic_graph_gnss_res.npz'

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
        est_opts = np.array([est_opts[0]])
        out_file = 'DEBUG'+out_file

    times = np.zeros(len(est_opts))
    pos_RMSEs = np.zeros(len(est_opts))
    ang_RMSEs = np.zeros(len(est_opts))

    # in_select and est_select control everything below
    for est_select in range(len(est_opts)):
        print('Running estimator',est_opts[est_select,0])
        # out_file = 'RMSE_input_'+in_opts[in_select,1]+'_est_'+est_opts[est_select,0]+'.npy'

        # Decide what cost function we will use for the measurements
        meas_noise = est_opts[est_select,1]

        in_data = get_chemnitz_data()
        if DEBUG:
            run_length = 500
        else:
            run_length = 4000 #len(in_data)
        ########   
        # Now run the optimziation (with whatever noise model you have)    
        start_time = time.time()
        np_est_poses = solve_scenario(in_data[:run_length], meas_noise=meas_noise)
        end_time = time.time()
        times[est_select] = end_time - start_time
        data_out_file = 'data_basic_graph_gnss_res_'+est_opts[est_select,0]+'.npz'
        if DEBUG:
            data_out_file = "DEBUG_"+data_out_file
        np.savez(data_out_file, est_states=np_est_poses)
        # plt.plot(np_est_poses)
        # plt.show()
        truth = np.array([in_data[i,1] for i in range(run_length)])


        if DEBUG:
            # When doing one run, good for plotting results
            fig = plt.figure()

            plt.plot(truth[:,0], truth[:,1])
            plt.plot(np_est_poses[:,0], np_est_poses[:,1])
            plt.legend(['truth', 'est'])
            plt.show()

        RMSE = m.sqrt(np.average(np.square(truth[:,:3]- np_est_poses[:,:3])))
        if DEBUG:
            print("RMSE (on x, y, and z) is",RMSE)

        pos_RMSEs[est_select] = RMSE
    # Required to load est_save back in from the .npz file
    est_save = est_opts[:,0].astype(str)
    np.savez(out_file, times=times, pos_RMSEs=pos_RMSEs, ang_RMSEs=ang_RMSEs, est_opts=est_save)


# %%
