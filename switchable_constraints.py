#%%
import gtsam
import numpy as np
import math as m
from tqdm import tqdm
from unicycle_est_utils import switchable_error_range_known_landmark, angleize_np_array, pose2_list_to_nparray, switchable_constraint_error
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
                   dyn_noise: gtsam.noiseModel = \
                    gtsam.noiseModel.Diagonal.Sigmas(np.array([.1,.1,.02]) * m.sqrt(.1)),
                   switch_noise: gtsam.noiseModel = \
                    gtsam.noiseModel.Isotropic.Sigma(1,.2)) \
                -> np.array:
    '''
    Take in a dictionary that has the 'measurements', 'inputs', 'x0', and 'landmarks' 
    locations in it.  Create a graph and return the optimized results as a np.array.  
    This graph is formulated such that each measurement (range measurement) also has
    a switchable constraint associated with it to detect outliers.  The switch_noise input
    is a Gaussian noise model for the switch constraint, i.e. how strong the bias to 
    the measurement being valid should (the smaller the noise, the stronger the bias)

    Inputs:
        input_meas_dict: the dictionary with the required information
        dt: the timestep between states (used with inputs to propagate)
        meas_noise:  What noise model to use with the measurements
        dyn_noise: What noise model to use with the dyanmics factors
        switch_noise: What noise model (weighting) to use with the switch constraint

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
    # Calculate the number of bits needed to represent the maximum number of landmarks
    landmark_bits = int(m.ceil(m.log2(nl)))
    # Lambda function to combine two integers by bit-shifting
    combine_integers = lambda top_int, bottom_int: (top_int << landmark_bits) | bottom_int
    # Switch key that associates a measurement with a landmark and then creates a "Key"
    switch_key = lambda l, x: gtsam.symbol(f's', combine_integers(x,l))

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

    # measurement (and switch) factors
    for ii,meas in enumerate(Z):
        for jj in range(nl):
            # Add measurement factor
            graph.add( gtsam.CustomFactor( meas_noise, [pose_key(ii), switch_key(jj,ii)], 
                                            partial(switchable_error_range_known_landmark, 
                                                    landmark_locs[jj], meas[jj] ) ) )
            # Add switching factor
            # graph.add( gtsam.CustomFactor( switch_noise, [switch_key(jj,ii)],
            #                                switchable_constraint_error) )
            graph.add( gtsam.PriorFactorDouble( switch_key(jj,ii), 1.0, switch_noise ) )

    # Graph is formed, but need some initial values for the poses and switches 
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

    # Also initialize the switchable constraint variable nodes
    for ii in range(len(Z)): # N+1, so not the same as previous loop :)
        for jj in range(nl):
            initial_estimates.insert( switch_key(jj,ii), 1.0 )

    ## Everything should be set up. Now to optimize
    ## TODO:  move to dogleg optimizer?
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

    if DEBUG:
        # To debug switching factors
        # First, get all the current values of the switching factors
        switch_values = np.zeros((len(Z),nl))
        for ii in range(len(Z)):
            for jj in range(nl):
                switch_values[ii,jj] = result.atDouble(switch_key(jj,ii))
        print('Switch Values are:\n',switch_values)
        plt.plot(np.sqrt(switch_values[:100].flatten()))
        plt.show()
                            


    return initial_np, np_est_poses


#%%
if __name__ == '__main__':
    out_file = 'switchable_constraints_unicycle_res.npz'
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
    est_opts = np.array([
        ['0.05', gtsam.noiseModel.Isotropic.Sigma(1,0.05)],
        ['0.1', gtsam.noiseModel.Isotropic.Sigma(1,0.1)],
        ['0.2', gtsam.noiseModel.Isotropic.Sigma(1,0.2)],
        ['0.3', gtsam.noiseModel.Isotropic.Sigma(1,0.3)],
        ['0.4', gtsam.noiseModel.Isotropic.Sigma(1,0.4)],
        ['0.5', gtsam.noiseModel.Isotropic.Sigma(1,0.5)]
    ])

    if DEBUG: # change this to know which one runs...
        in_opts = np.array([in_opts[0]])
        est_opts = np.array([est_opts[3]])
        which_run = 0
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
            print('Running input',in_opts[in_select,0], 'and estimator',est_opts[est_select,0])
            in_path = in_opts[in_select,1]
            # out_file = 'RMSE_input_'+in_opts[in_select,1]+'_est_'+est_opts[est_select,0]+'.npy'
            for ii in tqdm(run_list):
                # First, read in the data from the file
                in_file = in_path+f'run_{ii:04d}.npz'
                in_data = dict(np.load(in_file))

                in_data['x0'] = np.array([0, 0, m.pi/2])

                # Decide what cost function we will use for the switching factors
                switch_noise = est_opts[est_select,1]

                ########   
                # Now run the optimziation (with whatever noise model you have)    
                start_time = time.time()
                initial_np, np_est_poses = solve_scenario(in_data, switch_noise = switch_noise)
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
    est_save = est_opts[:,0].astype(str)
    in_opts_save = in_opts[:,0].astype(str)
    np.savez(out_file, times=times, pos_RMSEs=pos_RMSEs, ang_RMSEs=ang_RMSEs, in_opts=in_opts_save, est_opts=est_save)

# %%
