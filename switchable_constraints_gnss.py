#%%
import gtsam
import numpy as np
import math as m
from tqdm import tqdm
from gnss_est_utils import switchable_error_pseudorange, get_chemnitz_data, init_pos, error_clock, time_divider
from functools import partial
import copy
import time

DEBUG=False
if DEBUG:
    import matplotlib.pyplot as plt


def solve_scenario(gnss_data: np.array,
                   meas_noise: gtsam.noiseModel = \
                    gtsam.noiseModel.Isotropic.Sigma(1,1.),
                   dyn_Q_diags : np.array = np.array([4E-19, 1.4E-18]),
                   switch_noise: gtsam.noiseModel = \
                    gtsam.noiseModel.Isotropic.Sigma(1,.4)) \
                -> np.array:
    '''
    Take in a numpy array with all the data in it (see get_chemnitz_data for format),
    create a graph to handle it, and solve it.  The switch_noise input
    is a Gaussian noise model for the switch constraint, i.e. how strong the bias to 
    the measurement being valid should (the smaller the noise, the stronger the bias)

    Inputs:
        gnss_data:  the array with all the measurements and timestamps (and ground truth) in it
        meas_noise:  What noise model to use with the measurements
        dyn_noise: What noise model to use with the dyanmics factors
        switch_noise: What noise model (weighting) to use with the switch constraint

    Outputs:
        A Nx5 np.array with the output poses

    '''
    bcn_dyn_Q_diags = dyn_Q_diags * time_divider**2 # bcn = better condition number... Makes GTSAM happier
    # For a weak dynamics between positions, do 30 m/s, so running 60 miles/hour (approx) is only 1 sigma
    complete_dyn_Q_diags = np.ones(5)*900.
    complete_dyn_Q_diags[3:] = bcn_dyn_Q_diags
    # bcn_dyn_Q_diags[0] *=  time_divider**2 
    # Create the graph and optimize
    graph = gtsam.NonlinearFactorGraph()
    initial_estimates = gtsam.Values()

    ## Functions for creating keys -- identifiers for hidden variables
    pose_key = lambda x: gtsam.symbol('x',x)

    # Calculate the number of bits needed to represent the maximum number of satellites
    num_sats = np.array([len(gnss_data[ii,2]) for ii in range(len(gnss_data))])
    nl = np.max(num_sats)
    landmark_bits = int(m.ceil(m.log2(nl)))
    # Lambda function to combine two integers by bit-shifting
    combine_integers = lambda top_int, bottom_int: (top_int << landmark_bits) | bottom_int
    # Switch key that associates a measurement with a landmark and then creates a "Key"
    switch_key = lambda x, s: gtsam.symbol(f's', combine_integers(x,s))

    ## odometry factors.  Basically, no odometry except on the clock...
    for ii in range(1,len(gnss_data)):
        dt = gnss_data[ii,0]-gnss_data[ii-1,0]
        graph.add( gtsam.BetweenVector5Factor ( pose_key(ii-1), pose_key(ii), dt,
                                           gtsam.noiseModel.Diagonal.Variances(complete_dyn_Q_diags*dt) ) )
        # graph.add( gtsam.ClockErrorFactor ( pose_key(ii-1), pose_key(ii), dt,
        #                                    gtsam.noiseModel.Diagonal.Variances(bcn_dyn_Q_diags*dt) ) )
        # graph.add( gtsam.CustomFactor( gtsam.noiseModel.Diagonal.Variances(bcn_dyn_Q_diags*dt), 
        #                                [pose_key(ii-1), pose_key(ii)], 
        #                                partial(error_clock, dt) ) )
    # measurement (and switch) factors
    for ii,data in enumerate(gnss_data):
        meas_list = data[2]
        initial_estimates.insert( pose_key(ii), init_pos( meas_list ) )
        for jj in range(min(len(meas_list),12)):
            curr_meas = meas_list[jj]
            # Add measurement factor
            # graph.add( gtsam.CustomFactor( meas_noise, [pose_key(ii)],  
            #                                 partial(error_psuedorange, curr_meas[1], curr_meas[2:]) ) )
            graph.add( gtsam.sc_PseudoRangeFactor( pose_key(ii), switch_key(ii,jj), 
                                                  curr_meas[1], curr_meas[2:], meas_noise))
            # graph.add( gtsam.CustomFactor( meas_noise, [pose_key(ii), switch_key(ii,jj)], 
            #                                 partial(switchable_error_pseudorange, 
            #                                         curr_meas[2:], curr_meas[1]  ) ) ) # pass in satellite loc and pseudo-range
            # # Add switching factor
            # graph.add( gtsam.CustomFactor( switch_noise, [switch_key(ii,jj)],
            #                                switchable_constraint_error) )
            graph.add( gtsam.PriorFactorDouble( switch_key(ii,jj), 1.0, switch_noise ) )
            initial_estimates.insert( switch_key(ii,jj), 1.0 ) # initialize the switching constraint hidden variables

    ## Everything should be set up. Now to optimize
    ## TODO:  move to dogleg optimizer?
    parameters = gtsam.DoglegParams()
    parameters.setMaxIterations(50)
    parameters.setRelativeErrorTol(5E-4)
    parameters.setVerbosity("ERROR")
    optimizer = gtsam.DoglegOptimizer(graph, initial_estimates, parameters)
    result = optimizer.optimize()

    # Prepare the results to be returned.
    est_poses=np.zeros((len(gnss_data), 5))
    for ii in range(len(gnss_data)):
        est_poses[ii] = result.atVector(pose_key(ii))

    # if DEBUG:
    #     # To debug switching factors
    #     # First, get all the current values of the switching factors
    #     switch_values = np.zeros((len(Z),nl))
    #     for ii in range(len(Z)):
    #         for jj in range(nl):
    #             switch_values[ii,jj] = result.atDouble(switch_key(jj,ii))
    #     print('Switch Values are:\n',switch_values)
    #     plt.plot(np.sqrt(switch_values[:100].flatten()))
    #     plt.show()
                            
    return est_poses


#%%
if __name__ == '__main__':
    out_file = 'switchable_constraints_gnss_res.npz'

    # What weight to use on the switching model
    est_opts = np.array([
        ['SC-0.05', gtsam.noiseModel.Isotropic.Sigma(1,0.05)],
        ['SC-0.1', gtsam.noiseModel.Isotropic.Sigma(1,0.1)],
        ['SC-0.2', gtsam.noiseModel.Isotropic.Sigma(1,0.2)],
        ['SC-0.3', gtsam.noiseModel.Isotropic.Sigma(1,0.3)],
        ['SC-0.4', gtsam.noiseModel.Isotropic.Sigma(1,0.4)],
        ['SC-0.5', gtsam.noiseModel.Isotropic.Sigma(1,0.5)]
    ])

    if DEBUG: # change this to know which one runs...
        est_opts = np.array([est_opts[3]])
        out_file = 'DEBUG'+out_file

    times = np.zeros(len(est_opts))
    pos_RMSEs = np.zeros(len(est_opts))

    # This is considered constant for all these runs:
    meas_noise = gtsam.noiseModel.Isotropic.Sigma( 1, 1.0 )

    # in_select and est_select control everything below
    for est_select in range(len(est_opts)):
        print('Running estimator',est_opts[est_select,0])

        # Decide what cost function we will use for the switching factors
        switch_noise = est_opts[est_select,1]

        in_data = get_chemnitz_data()
        if DEBUG:
            run_length = 4
        else:
            run_length = len(in_data)
        ########   
        # Now run the optimziation (with whatever noise model you have)    
        start_time = time.time()
        np_est_poses = solve_scenario(in_data[:run_length], switch_noise=switch_noise)
        end_time = time.time()
        times[est_select] = end_time - start_time
        data_out_file = 'data_swichable_constraints_gnss_res_'+est_opts[est_select,0]+'.npz'
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


        RMSE = m.sqrt(np.average(np.square(truth[:,:2]- np_est_poses[:,:2])))
        if DEBUG:
            print("RMSE (on x, y, and z) is",RMSE)
        pos_RMSEs[est_select] = RMSE
        
    est_save = est_opts[:,0].astype(str)
    np.savez(out_file, times=times, pos_RMSEs=pos_RMSEs, est_opts=est_save)

# %%
