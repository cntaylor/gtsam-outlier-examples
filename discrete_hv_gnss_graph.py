#%%
import gtsam
import numpy as np
import numpy.linalg as la
import math as m
from tqdm import tqdm
from gnss_est_utils import time_divider, c_small, get_chemnitz_data, init_pos
from functools import partial
import copy
import time
# from numba import jit

DEBUG=True
if DEBUG:
    import matplotlib.pyplot as plt

def solve_scenario(gnss_data: np.array,
                   meas_noise: gtsam.noiseModel = \
                    gtsam.noiseModel.Isotropic.Sigma(1,1.),
                   outlier_noise: gtsam.noiseModel = \
                    gtsam.noiseModel.Isotropic.Sigma(1,1000.),
                   dyn_Q_diags : np.array = np.array([4E-19, 1.4E-18]) ) \
                -> tuple[np.array, np.array]:
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
        A tuple with:
         (1) Nx5 np.array with the output poses
         (2) Nxmax_sats np.array with binary values whether outlier (true) or not

    '''
    bcn_dyn_Q_diags = dyn_Q_diags * time_divider**2 # bcn = better condition number... Makes GTSAM happier
    # For a weak dynamics between positions, do 30 m/s, so running 60 miles/hour (approx) is only 1 sigma
    complete_dyn_Q_diags = np.ones(5)*900.
    complete_dyn_Q_diags[3:] = bcn_dyn_Q_diags

    ## Functions for creating keys -- identifiers for hidden variables
    pose_key = lambda x: gtsam.symbol('x',x)

    # Calculate the number of bits needed to represent the maximum number of satellites
    num_sats = np.array([len(gnss_data[ii,2]) for ii in range(len(gnss_data))])
    nl = np.max(num_sats)

    # Array for holding outlier discrete values
    outlier_bools = np.empty((len(gnss_data),nl), dtype=bool)

    def compute_prob(diff: float, cov : float) -> float:
        # Compute the probability according to the full normal distribution
        # MUST include the scaling factor!
        return m.exp(-.5*diff**2/cov ) * 1/m.sqrt(2*m.pi*cov)
    
    def compute_prob_full(meas : np.array, meas_cov: float, vec_est : np.array, vec_inf : np.array) -> float:
        # Compute the probability by removing the effect of the measurement
        # from the current estimated covariance, then compute the full probability
        # First, remove the effect of the measurement from the information matrix
        # Note that vec_inf is assumed to be 4x4 as I ignore clock rate error 
        diff_loc = vec_est[:3] - meas[2:]
        diff_loc_uv = diff_loc / la.norm(diff_loc) # uv = unit vector
        full_deriv = np.array([diff_loc_uv[0], diff_loc_uv[1], diff_loc_uv[2], c_small])
        meas_proj = full_deriv / la.norm(full_deriv)
        inf_removed = vec_inf - np.outer(meas_proj, meas_proj)/meas_cov
        cov_from_est = meas_proj @ la.inv(inf_removed) @ meas_proj
        pred_meas = la.norm(diff_loc) + c_small * vec_est[3]
        return compute_prob(pred_meas - meas[1], cov_from_est+meas_cov)

    def form_continuous_graph(gnss_data : np.array,
                        outlier_bools : np.array,
                        meas_noise : gtsam.noiseModel,
                        outlier_noise : gtsam.noiseModel,
                        dyn_noise : np.array ) -> gtsam.NonlinearFactorGraph:
        '''
        Take in the measurements, current values, outlier values, and create a graph and return it
        This graph will have different measurement noises associated with each measurement depending on the
        values in outlier_vals

        Inputs:
            gnss_data: the array with all the measurements and timestamps in it
            outlier_vals: Which measurements are considered outliers (an N x nl array of bools)
            meas_noise:  What noise model to use with the measurements when they are not outliers
            outlier_noise: What noise model to use with the measurements when they are outliers
            dyn_noise: What noise model to use with the dyanmics factors

        Outputs:
            The graph (gtsam.NonlinearFactorGraph)
        '''

        # Create the graph
        graph = gtsam.NonlinearFactorGraph()

        ## odometry factors.  Basically, no odometry except on the clock 
        ## (and a really weak prior between two locations in case all the satellites get "outliered" away)
        for ii in range(1,len(gnss_data)):
            dt = gnss_data[ii,0]-gnss_data[ii-1,0]
            graph.add( gtsam.BetweenVector5Factor ( pose_key(ii-1), pose_key(ii), dt,
                                            gtsam.noiseModel.Diagonal.Variances(dyn_noise*dt) ) )
            # graph.add( gtsam.ClockErrorFactor ( pose_key(ii-1), pose_key(ii), dt,
            #                                    gtsam.noiseModel.Diagonal.Variances(bcn_dyn_Q_diags*dt) ) )
            # graph.add( gtsam.CustomFactor( gtsam.noiseModel.Diagonal.Variances(bcn_dyn_Q_diags*dt), 
            #                                [pose_key(ii-1), pose_key(ii)], 
            #                                partial(error_clock, dt) ) )
        # measurement (and switch) factors
        for ii,data in enumerate(gnss_data):
            meas_list = data[2]
            for jj in range(len(meas_list)):
                curr_meas = meas_list[jj]
                # Add measurement factor
                # graph.add( gtsam.CustomFactor( meas_noise, [pose_key(ii)],  
                #                                 partial(error_psuedorange, curr_meas[1], curr_meas[2:]) ) )
                if outlier_bools[ii,jj]:
                    graph.add( gtsam.PseudoRangeFactor( pose_key(ii),
                                                    curr_meas[1], curr_meas[2:], outlier_noise))
                else:
                    graph.add( gtsam.PseudoRangeFactor( pose_key(ii), 
                                                    curr_meas[1], curr_meas[2:], meas_noise))

        return graph
        
    # Compute initial values using all measurements (no outlier rejection)
    initial_estimates = gtsam.Values()
    initial_locs = np.zeros((len(gnss_data),5))
    for ii,data in enumerate(gnss_data):
        meas_list = data[2]
        loc_cov = np.empty((4,4), dtype=float)
        initial_locs[ii] = init_pos(meas_list, loc_cov)
        initial_estimates.insert( pose_key(ii), initial_locs[ii] )
        loc_inf = la.inv(loc_cov)
        # Go through and initialize the outlier discrete values as well
        for jj,meas in enumerate(meas_list):
            
            # Compute whether each one should be an inlier or an outlier
            inlier_prob = compute_prob_full( meas, meas_noise.covariance().item(), initial_locs[ii], loc_inf)
            outlier_prob = compute_prob_full( meas, outlier_noise.covariance().item(), initial_locs[ii], loc_inf)
            outlier_bools[ii,jj] = outlier_prob > inlier_prob
    # Everything should be set up. Now to optimize
    ## This is essentially an EM (expectation maximization) algorithm.
    ## First, (create and) optimize the continuous graph for the current discrete values
    ## Then, set the discrete values given the new continuous values.
    ## Rinse and repeat...
    parameters = gtsam.GaussNewtonParams()
    parameters.setVerbosity("ERROR")
    curr_values = initial_estimates
    outlier_not_changed_count = 0
    num_loops = 0
    ############# The Optimization Loop #############
    # This is the EM loop
    while num_loops < 35 and (not outlier_not_changed_count==2): # optimization iterations ... a stupid, but simple way to start
        # Optimize the continuous graph
        if outlier_not_changed_count == 1:  # if the outlier values have not changed, let the continuous optimize more
            parameters.setMaxIterations(20)
        else:
            parameters.setMaxIterations(2)
        graph = form_continuous_graph(gnss_data, outlier_bools, meas_noise, outlier_noise, complete_dyn_Q_diags)
        optimizer = gtsam.GaussNewtonOptimizer(graph, curr_values, parameters)
        curr_values = optimizer.optimize()
        # Set the discrete values
        ## Need to figure out what the covariance at each pose is
        ## Take covariance at each pose, invert it (to get the information matrix),
        ## subtract out information due to the measurement being evaluated for being an outlier.
        ## The compute the probability for the difference between estimated and measured values
        ## given a covariance of the inlier or outlier model + the information matrix computed above (and then inverted)
        marginals = gtsam.Marginals(graph, curr_values)
        old_outliers = outlier_bools.copy()
        for ii,data in enumerate(gnss_data):
            meas_list = data[2]
            curr_cov = marginals.marginalCovariance(pose_key(ii))[:4,:4]
            raw_info = la.inv(curr_cov)
            curr_loc = curr_values.atVector(pose_key(ii))
            for jj,meas in enumerate(meas_list):
                # Compute whether each one should be an inlier or an outlier
                inlier_prob = compute_prob_full(meas, meas_noise.covariance().item(), curr_loc, raw_info)
                outlier_prob = compute_prob_full(meas, outlier_noise.covariance().item(), curr_loc, raw_info)
                outlier_bools[ii,jj] = outlier_prob > inlier_prob
        if np.array_equal(outlier_bools,old_outliers):
            outlier_not_changed_count +=1
        else:
            outlier_not_changed_count = 0
        num_loops += 1
        if DEBUG:
            print('num_loops is',num_loops, 'and outlier_not_changed_count is', outlier_not_changed_count)
            print('Number of outliers is', np.sum(outlier_bools))
        
    


    # Prepare the results to be returned.
    est_poses=np.zeros((len(gnss_data),3))
    for ii in range(len(gnss_data)):
        est_poses[ii]=(curr_values.atVector(pose_key(ii))[:3])

    return initial_locs, est_poses


if __name__ == '__main__':
    out_file = 'discrete_independent_gnss_res.npz'

    # What weight to use on the switching model
    est_opts = np.array([
        ['DI', gtsam.noiseModel.Isotropic.Sigma(1,1000.)]
    ])

    if DEBUG: # change this to know which one runs...
        est_opts = np.array([est_opts[0]])
        out_file = 'DEBUG'+out_file

    times = np.zeros(len(est_opts))
    pos_RMSEs = np.zeros(len(est_opts))

    # This is considered constant for all these runs:
    meas_noise = gtsam.noiseModel.Isotropic.Sigma( 1, 1.0 )

    # in_select and est_select control everything below
    for est_select in range(len(est_opts)):
        print('Running estimator',est_opts[est_select,0])

        # Decide what cost function we will use for the switching factors
        outlier_noise = est_opts[est_select,1]

        in_data = get_chemnitz_data()
        if DEBUG:
            run_length = 10
        else:
            run_length = len(in_data)
        ########   
        # Now run the optimziation (with whatever noise model you have)    
        start_time = time.time()
        init_poses, np_est_poses = solve_scenario(in_data[:run_length], outlier_noise=outlier_noise)
        end_time = time.time()
        times[est_select] = end_time - start_time
        data_out_file = 'data_discrete_ind_gnss_res_'+est_opts[est_select,0]+'.npz'
        if DEBUG:
            data_out_file = "DEBUG_"+data_out_file
        np.savez(data_out_file, est_states=np_est_poses, init_poses = init_poses)

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
