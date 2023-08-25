'''
This creates and the estimates a simple unicycle robot 
(Pose2 type thing) with a ranging measurement to two 
landmarks of known location and a simple dynamics model
where V (velocity) and w (turn rate) are the inputs
'''

#%%
import math as m
import numpy as np
import gtsam
import matplotlib.pyplot as plt
import copy
import examples as ef #example functions

#For each noise, I make raw (sqrt) values (for simulation) followed by the 
# Noise factor. I use the sqrt style matrix where S0 * S0.T gives
#the full covariance matrix.  This also means that S0 * a vector sample of isometric 
# normally distributed values gives you a sample from P0

#!!!! Note that for now, I assume everything is diagonal, so S0 is a bit overkill!!!!
s_P = np.array([1.,1.,m.pi/8])
S0 = np.diag(s_P)
prior_noise = gtsam.noiseModel.Diagonal.Sigmas(s_P)

s_Q = np.array([.1,.1,.01])
SQ = np.diag(s_Q)
process_noise = gtsam.noiseModel.Diagonal.Sigmas(s_Q)

SR = np.diag(np.array([1.]))
meas_noise = gtsam.noiseModel.Diagonal.Sigmas(SR)

dt = .1
#Set up velocity commands
V_one = int(round(5/dt)) #steps to run at 1m/s
V_two = int(round(3/dt)) # steps to run at 2 m/s
V_command = np.append(np.ones(V_one),2*np.ones(V_two))
#Set up rotation commands
w_straight =int(round(5/dt)) # steps to go straight
seconds_to_turn=3
w_turn = int(round(seconds_to_turn/dt)) #steps to turn
#pi/(2*...) is to make it turn 90 degrees in the turning time. Can modify (obviously)
w_command = np.append(np.zeros(w_straight),np.ones(w_turn)*m.pi/(2*seconds_to_turn))

# Create Landmarks and functions for creating keys
nl = 2 # number of landmarks
lm_key = lambda x: x
pose_key = lambda x: x+nl

landmark_locs = np.random.rand(nl,2)*50. - 25

landmarks =[]
for curr_lm in landmark_locs:
    landmarks.append(gtsam.Point2(curr_lm))

## Generate the truth data and measurements first
N = round(60/dt) #number of timesteps forward (there will be N+1 poses)

x0 = gtsam.Pose2(pose_key(0),0,m.pi/2.)

true_poses = []
initial_error = S0.dot(np.random.randn(3))
curr_x = ef.dynamics(x0,0,0,0,add_noise=initial_error)
true_poses.append(curr_x)

#While looping through, create measurements as well
measurements = np.zeros((N,nl))

for ii,curr_meas in enumerate(measurements):
    proc_error = SQ.dot(np.random.randn(3))
    curr_x = ef.dynamics(curr_x,ef.modulo_idx(V_command,ii), ef.modulo_idx(w_command,ii), dt, add_noise=proc_error)
    true_poses.append(curr_x)
    # while here, do measurements
    for jj,lm in enumerate(landmarks):
        curr_meas[jj] = curr_x.range(lm) + SR.dot(np.random.randn(1))

## Create the graph and optimize
# I am going to do the keys as follows.  Positive integers are locations (Pose2)
# of the object, where the key corresponds with the timestep.  So, 0->N.  Negative
# numbers correspond with landmarks (-1 is first landmark, -2 is second, etc.)
graph = gtsam.NonlinearFactorGraph()

#prior
graph.add(gtsam.PriorFactorPose2(pose_key(0), x0, prior_noise))

# odometry factors
for ii in range(N):
    curr_V = ef.modulo_idx(V_command,ii) * dt
    curr_w = ef.modulo_idx(w_command,ii) * dt
    Vx = curr_V * m.cos(curr_w/2.)
    Vy = curr_V * m.sin(curr_w/2.)
    graph.add(gtsam.BetweenFactorPose2(pose_key(ii),pose_key(ii+1),gtsam.Pose2(Vx,Vy,curr_w),process_noise))

# measurement factors
for ii,meas in enumerate(measurements):
    for jj in range(nl):
        graph.add(gtsam.RangeFactor2D(pose_key(ii+1),lm_key(jj),meas[jj],meas_noise))

#Now let's initialize the graph
initial_estimates = gtsam.Values()
#Landmarks at fixed locations... Lock them in the graph as well...
for jj,lm in enumerate(landmarks):
    initial_estimates.insert(lm_key(jj), lm )
    graph.add(gtsam.NonlinearEqualityPoint2(lm_key(jj),copy.copy(lm))) #Not sure if copy.copy is really needed, but to be sure :)

#Use odometry only to initialize the graph
initial_estimates.insert(pose_key(0),x0)
curr_x=x0
initial_np = np.zeros((N+1,3))
initial_np[0] = np.array([curr_x.x(), curr_x.y(), curr_x.theta()])
for ii in range(N):
    curr_x = ef.dynamics(curr_x,ef.modulo_idx(V_command,ii), ef.modulo_idx(w_command,ii), dt)
    initial_estimates.insert(pose_key(ii+1), curr_x)
    initial_np[ii+1] = np.array([curr_x.x(), curr_x.y(), curr_x.theta()])

## Everything should be set up. Now to optimize
parameters = gtsam.GaussNewtonParams()
parameters.setMaxIterations(100)
parameters.setVerbosity("ERROR")
optimizer = gtsam.GaussNewtonOptimizer(graph, initial_estimates, parameters)
result = optimizer.optimize()

#%%  Plot the results
print('Landmarks are',landmark_locs)
est_poses=[]
for ii in range(N):
    est_poses.append(result.atPose2(pose_key(ii)))
est_poses.append(result.atPose2(pose_key(N)))
np_est_poses = ef.pose2_list_to_nparray(est_poses)

np_true_poses = ef.pose2_list_to_nparray(true_poses)

fig = plt.figure()

plt.plot(np_true_poses[:,0],np_true_poses[:,1])
plt.plot(np_est_poses[:,0],np_est_poses[:,1])
plt.plot(initial_np[:,0],initial_np[:,1])
plt.legend(['truth', 'est', 'initial'])
plt.show()
