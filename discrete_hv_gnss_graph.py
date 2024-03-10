#%%
import gtsam
import numpy as np
import numpy.linalg as la
import math as m
from tqdm import tqdm
from unicycle_est_utils import error_range_known_landmark, angleize_np_array, pose2_list_to_nparray
from functools import partial
import copy
import time

DEBUG=False
if DEBUG:
    import matplotlib.pyplot as plt


    
def solve_scenario(in_data : dict, 
                   dt : float = .1,
                   meas_noise: gtsam.noiseModel = \
                    gtsam.noiseModel.Isotropic.Sigma(1,1.),
                   outlier_noise: gtsam.noiseModel = \
                    gtsam.noiseModel.Isotropic.Sigma(1,1000.),
                   dyn_noise: gtsam.noiseModel = \
                    gtsam.noiseModel.Diagonal.Sigmas(np.array([.1,.1,.02]) * m.sqrt(.1)) ) \
                -> np.array:
    '''
    Take in a dictionary that has the 'measurements', 'inputs', 'x0', and 'landmarks' 
    locations in it.  Create a graph and return the optimized results as a np.array.  
    To solve the discrete, I will have to form a graph multiple times with different
    noise factors depending on what the discrete values are.  Each measurement has a 
    discrete variable that decides whether it will be an outlier or not.

    Inputs:
        input_meas_dict: the dictionary with the required information
        dt: the timestep between states (used with inputs to propagate)
        meas_noise:  What noise model to use with the measurements
        outlier_noise: What noise model to use with the outliers
        dyn_noise: What noise model to use with the dyanmics factors

    Outputs:
        A Nx3 np.array with the output poses

    '''

    Z = in_data['measurements'] # Z is the set of all measurements
    N = len(Z)
    U = in_data['inputs'] # U is the set of all inputs, should be length N (which had -1 to get it)
    assert len(U)==N-1, "inputs and measurements have incompatible length"
    landmark_locs = in_data['landmarks']

    x0 = in_data['x0']

    ## Functions for creating keys -- identifiers for hidden variables
    nl = len(landmark_locs) # number of landmarks
    pose_key = lambda x: gtsam.symbol('x',x)
    # Array for holding outlier discrete values
    outlier_bools = np.empty((N,nl), dtype=bool)

    def compute_prob(diff: float, cov : float) -> float:
        # Compute the probability according to the full normal distribution
        # MUST include the scaling factor!
        return m.exp(-.5*diff**2/cov ) * 1/m.sqrt(2*m.pi*cov)

    def form_continuous_graph(Z : np.array,
                        U : np.array,
                        outlier_vals : np.array,
                        landmark_locs : np.array,
                        meas_noise : gtsam.noiseModel,
                        outlier_noise : gtsam.noiseModel,
                        dyn_noise : gtsam.noiseModel ) -> gtsam.NonlinearFactorGraph:
        '''
        Take in the measurements, inputs, current values, outlier values, and create a graph and return it
        This graph will have different measurement noises associated with each measurement depending on the
        values in outlier_vals

        Inputs:
            Z: the measurements (an N x nl array) where N is the number of time steps and nl is the number of landmarks
            U: the inputs (an N-1 x 2 array) 
            outlier_vals: Which measurements are considered outliers (an N x nl array of bools)
            landmark_locs: the locations of the landmarks (an nl x 2 array)
            meas_noise:  What noise model to use with the measurements when they are not outliers
            outlier_noise: What noise model to use with the measurements when they are outliers
            dyn_noise: What noise model to use with the dyanmics factors

        Outputs:
            The graph (gtsam.NonlinearFactorGraph)
        '''

        # Create the graph
        graph = gtsam.NonlinearFactorGraph()

        ## odometry factors
        ### Note that this uses Lie Algebra sorts of things. The factor is the difference between the 
        ### current and previous pose, in the the previous pose's coordinate frame.  So, it is a 
        ### differentiable Pose factor
        for ii in range(N-1):
            curr_V = U[ii,0] * dt
            curr_w = U[ii,1] * dt
            Vx = curr_V * m.cos(curr_w/2.)
            Vy = curr_V * m.sin(curr_w/2.)
            graph.add( gtsam.BetweenFactorPose2( pose_key(ii), pose_key(ii+1), gtsam.Pose2( Vx, Vy, curr_w ), dyn_noise ) )

        # measurement factors
        for ii,meas in enumerate(Z):
            for jj in range(nl):
                if outlier_vals[ii,jj]:
                    # Add measurement factor
                    graph.add( gtsam.CustomFactor( outlier_noise, [pose_key(ii)], 
                                                    partial(error_range_known_landmark, landmark_locs[jj], meas[jj] ) ) )
                else:
                    graph.add( gtsam.CustomFactor( meas_noise, [pose_key(ii)],
                                                    partial(error_range_known_landmark, landmark_locs[jj], meas[jj] ) ) )
        
        return graph
        
    # Compute initial values using only odometry. Store the initial estimate as well for plotting (initial_np)
    # Also, initialize the outlier values for the odometery and measurement values.
    initial_estimates = gtsam.Values()
    initial_estimates.insert( pose_key(0), gtsam.Pose2(*x0) )
    curr_x=copy.copy( x0 )
    initial_np = np.zeros((N,3))
    initial_np[0] = x0
    for ii in range(N-1):
        curr_x[0] += U[ii,0]*dt * m.cos(curr_x[2]+U[ii,1]*dt/2.)
        curr_x[1] += U[ii,0]*dt * m.sin(curr_x[2]+U[ii,1]*dt/2.)
        curr_x[2] += U[ii,1]*dt
        initial_estimates.insert(pose_key(ii+1), gtsam.Pose2( *curr_x ) )
        initial_np[ii+1] = curr_x

    # Go through and initialize the outlier discrete values as well
    for ii,meas in enumerate(Z):
        for jj in range(nl):
            est_dist = np.linalg.norm(landmark_locs[jj] - initial_np[ii,:2])
            cov_proj = (landmark_locs[jj] - initial_np[ii,:2])/est_dist
            # This covariance is not too accurate, but hopefully better than nothing
            marg_cov = cov_proj @ (dyn_noise.covariance()[:2,:2] * ii) @ cov_proj
            inlier_prob = compute_prob(est_dist-meas[jj], meas_noise.covariance().item()+marg_cov)
            outlier_prob = compute_prob(est_dist-meas[jj], outlier_noise.covariance().item()+marg_cov)
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
    # TODO: Figure out a better way to run the iterations, converging when things don't change
    while num_loops < 20 and (not outlier_not_changed_count==2): # optimization iterations ... a stupid, but simple way to start
        # Optimize the continuous graph
        if outlier_not_changed_count == 1:  # if the outlier values have not changed, let the continuous optimize more
            parameters.setMaxIterations(20)
        else:
            parameters.setMaxIterations(2)
        graph = form_continuous_graph(Z, U, outlier_bools, landmark_locs, meas_noise, outlier_noise, dyn_noise)
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
        for ii,meas in enumerate(Z):
            # Find the marginal covariance for each pose
            curr_cov =marginals.marginalCovariance(pose_key(ii))[:2,:2]
            # If the marginal is in the "manifold" space, then I need to rotate it to a global coordinate frame
            theta = curr_values.atPose2(pose_key(ii)).theta()
            DCM = np.array([[m.cos(theta), -m.sin(theta)], [m.sin(theta), m.cos(theta)]])
            curr_cov = DCM @ curr_cov @ DCM.T
            raw_info = la.inv(curr_cov)

            for jj in range(nl):
                # Compute the contribution to the information matrix from the current measurement and remove it
                diff_loc = landmark_locs[jj] - curr_values.atPose2(pose_key(ii)).translation()
                pred_meas = la.norm(diff_loc)
                inf_vect = diff_loc / pred_meas
                cov_scale = outlier_noise.sigma() if outlier_bools[ii,jj] else meas_noise.sigma()
                pose_cov = inf_vect @ la.inv(raw_info - cov_scale**(-2) * np.outer(inf_vect, inf_vect) ) @ inf_vect
                if DEBUG:
                    print('ii is',ii,'and jj is',jj, 'and pose_cov is',pose_cov)
                    print('raw_info is',raw_info)
                    print('cov_scale is',cov_scale, 'mult by inf_vect is',cov_scale**(-2) * np.outer(inf_vect, inf_vect) )
                    print('la.inv part is',la.inv(raw_info - cov_scale**(-2) * np.outer(inf_vect, inf_vect) ))

                # Compute the probability of the difference between the estimated and measured value
                inlier_prob = compute_prob(pred_meas-meas[jj], meas_noise.covariance().item() + pose_cov)
                outlier_prob = compute_prob(pred_meas-meas[jj], outlier_noise.covariance().item() + pose_cov)
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
    est_poses=[]
    for ii in range(N):
        est_poses.append(curr_values.atPose2(pose_key(ii)))
    np_est_poses = pose2_list_to_nparray(est_poses)

    return initial_np, np_est_poses


#%%
if __name__ == '__main__':
    out_file = 'discrete_hv_unicycle_res.npz'
    n_runs = 100
    # This is a data structure that holds the directory name and
    # what the output file should say so they get picked together!
    in_opts = np.array([
        ['No outliers', 'no_outliers/'],
        ['10% outliers', 'measurement_10pc_outliers/'],
        ['20% outliers', 'measurement_20pc_outliers/'],
        ['30% outliers', 'measurement_30pc_outliers/'],
        ['40% outliers', 'measurement_40pc_outliers/'],
        ['50% outliers', 'measurement_50pc_outliers/'],
    ])

    # What weight to use on the switching model
    est_opts = np.array([ 'DI']) # DI = discrete independent

    if DEBUG: # change this to know which one runs...
        in_opts = np.array([in_opts[0]])
        est_opts = np.array([est_opts[0]])
        which_run = 21
        run_list=[which_run]
        out_file = 'DEBUG'+out_file
    else:
        run_list = np.arange(n_runs)

    times = np.zeros((len(in_opts),len(est_opts),n_runs))
    pos_RMSEs = np.zeros((len(in_opts),len(est_opts),n_runs))
    ang_RMSEs = np.zeros((len(in_opts),len(est_opts),n_runs))

    # This is considered constant for all these runs:
    meas_noise = gtsam.noiseModel.Isotropic.Sigma( 1, 1.0 )

    # in_select and est_select control everything below
    for in_select in range(len(in_opts)):
        for est_select in range(len(est_opts)):
            print('Running input',in_opts[in_select,0], 'and estimator',est_opts[est_select])
            in_path = in_opts[in_select,1]
            # out_file = 'RMSE_input_'+in_opts[in_select,1]+'_est_'+est_opts[est_select,0]+'.npy'
            for ii in tqdm(run_list):
                # First, read in the data from the file
                in_file = in_path+f'run_{ii:04d}.npz'
                in_data = dict(np.load(in_file))

                in_data['x0'] = np.array([0, 0, m.pi/2])

                ########   
                # Now run the optimziation (with whatever noise model you have)    
                start_time = time.time()
                initial_np, np_est_poses = solve_scenario(in_data)
                end_time = time.time()
                times[in_select,est_select,ii] = end_time - start_time

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
                pos_RMSEs[in_select,est_select,ii] = RMSE
                ang_RMSEs[in_select,est_select,ii] = RMSE_ang
            # print("Average RMSEs (pos & angle) are",np.average(RMSEs,1))
            # plt.plot(RMSEs)
            # plt.show()
    est_save = est_opts.astype(str)
    in_opts_save = in_opts[:,0].astype(str)
    np.savez(out_file, times=times, pos_RMSEs=pos_RMSEs, ang_RMSEs=ang_RMSEs, in_opts=in_opts_save, est_opts=est_save)

# %%
