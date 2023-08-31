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
import examples as ef #example functions

def pose2_list_to_nparray(pose2_list):
    going_out = np.zeros((len(pose2_list),3))
    for ii,cp in enumerate(pose2_list):
        going_out[ii] = np.array([cp.x(),cp.y(),cp.theta()])
    return going_out

def angelize_np_array(np_array):
    ''' 
    Take in an np array and make them all valid angles (between -pi and pi) by adding
    and subtracting 2*pi values to them
    '''
    while np.any(np_array < -m.pi):
        np_array[np_array < -m.pi] += 2 * m.pi
    while np.any(np_array > m.pi):    
        np_array[np_array > m.pi] -= 2 * m.pi
    return np_array

# TODO:  take most of main and make it a function, optimize, that takes in Z and U and 
# returns the est_state.  Then RMSE computations can be done outside, enabling
# Monte Carlo experiments with this code

if __name__ == '__main__':
    # First, read in the data from the file
    in_file = 'unicycle_data.npz'
    in_data = np.load(in_file)
    Z = in_data['measurements'] # Z is the set of all measurements
    N = len(Z) - 1
    U = in_data['inputs'] # U is the set of all inputs, should be length N (which had -1 to get it)
    assert len(U)==N, "inputs and measurements have incompatible length"
    landmark_locs = in_data['landmarks']

    # The next few lines of code will work best when it matches how the system was run
    # Up to the user to make sure this matches with unicycle_sim output.
    dt = .1

    s_Q = np.array([.1,.1,.02]) * m.sqrt(dt)
    process_noise = gtsam.noiseModel.Diagonal.Sigmas(s_Q)

    SR = np.diag(np.array([1.]))
    meas_noise = gtsam.noiseModel.Diagonal.Sigmas(SR)

    x0 = np.array([0, 0, m.pi/2])

    # Create the graph and optimize
    graph = gtsam.NonlinearFactorGraph()
    initial_estimates = gtsam.Values()

    ## I am going to do the keys as follows. 0->(nl-1) are the keys for the landmarks.
    ## nl -> N+nl-1 will be the unicycle locations
    ## (See comment below on why we need keys for the landmark locations, even though they are known)

    ## Functions for creating keys
    nl = len(landmark_locs) # number of landmarks
    lm_key = lambda x: x
    pose_key = lambda x: x+nl

    ## To create "range" measurements between the landmark and the robot, need to
    ## have landmark hidden variables.  Don't really want to estimate their locations
    ## though, so put in an equality constraint to their location

    ## Landmarks at fixed locations... Lock them in the graph as well...
    for jj,curr_lm in enumerate(landmark_locs):
        lm = gtsam.Point2(curr_lm)
        initial_estimates.insert(lm_key(jj), lm )
        graph.add(gtsam.NonlinearEqualityPoint2(lm_key(jj), lm)) 


    ## odometry factors
    ### Note that this uses Lie Algebra sorts of things. The factor is the difference between the 
    ### current and previous pose, in the the previous pose's coordinate frame.  So, it is a 
    ### differentiable Pose factor
    for ii in range(N):
        curr_V = U[ii,0] * dt
        curr_w = U[ii,1] * dt
        Vx = curr_V * m.cos(curr_w/2.)
        Vy = curr_V * m.sin(curr_w/2.)
        graph.add( gtsam.BetweenFactorPose2( pose_key(ii), pose_key(ii+1), gtsam.Pose2( Vx, Vy, curr_w ), process_noise ) )

    # measurement factors
    for ii,meas in enumerate(Z):
        for jj in range(nl):
            graph.add( gtsam.RangeFactor2D( pose_key(ii), lm_key(jj), meas[jj], meas_noise ) )

    ## Graph is formed, but need some initial values for the 
    ## Use odometry only to initialize the graph.  Store the initial estimate as well for plotting (initial_np)
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

    #%%  Plot the results
    est_poses=[]
    for ii in range(N):
        est_poses.append(result.atPose2(pose_key(ii)))
    est_poses.append(result.atPose2(pose_key(N)))
    np_est_poses = pose2_list_to_nparray(est_poses)
    truth = in_data['truth']
    RMSE = m.sqrt(np.average(np.square(truth[:,:2]- np_est_poses[:,:2])))
    print("RMSE (on x and y) is",RMSE)
    RMSE_ang = m.sqrt(np.average( angelize_np_array( np.square(truth[:,2]- np_est_poses[:,2]) ) ) )
    print("RMSE (on angle) is",RMSE_ang)

    fig = plt.figure()

    plt.plot(truth[:,0], truth[:,1])
    plt.plot(np_est_poses[:,0], np_est_poses[:,1])
    plt.plot(initial_np[:,0], initial_np[:,1])
    plt.legend(['truth', 'est', 'initial'])
    plt.show()
