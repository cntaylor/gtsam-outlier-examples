'''
The point of this Python script is to create a file that holds the
1.  true pose of the unicycle at all times, 
2.  the location of landmarks to which a ranging measurement is made at each timestep, 
3.  the corrupted range measurements that are obtained.
4.  the inputs that control the unicycle over time

How the true pose moves and measurements are corrupted are specified by which functions
are actually called.  Some possible functions are also found in this file
'''

import numpy as np
from math import pi, sqrt, cos, sin
from functools import partial
from copy import copy

def angle_bound_rad(in_angle : float) -> float:
    # Simple check to put the value between -pi and pi
    going_out=in_angle
    if in_angle < -pi:
        going_out += 2*pi
    if in_angle > pi:
        going_out -= 2*pi
    return going_out

#######
# All process corruption functions will take in a pose (as a 3-element numpy array) and 
# add noise to it. Output is a 3-element pose.  If more inputs are needed, these should be
# bound using partial
def noiseless_dyn(curr_state: np.array, inputs : np.array, dt : float) -> np.array:
    move_dir = curr_state[2] + inputs[1]*dt/2
    going_out=np.zeros(3)
    going_out[2] = angle_bound_rad( curr_state[2] + inputs[1]*dt )
    going_out[0] = curr_state[0] + inputs[0]*dt*cos(move_dir)
    going_out[1] = curr_state[1] + inputs[0]*dt*sin(move_dir)
    return going_out

def process_white_noise(next_state : np.array, S_Q: np.array) -> np.array:
    out_state = next_state + S_Q@np.random.randn(3)
    out_state[2]= angle_bound_rad(out_state[2])
    return out_state
    
#######
# All measurement corruption functions will take in a 'nl' element numpy array and return
# a corrupted numpy array

def noiseless_meas(state : np.array, landmark_locs : np.array) -> np.array:
    tmp =landmark_locs - state[:2]
    return np.sqrt( np.sum( np.square( tmp ), axis=1 ) )

def measurement_isotropic_noise(measurements : np.array, S_R : float) -> np.array:
    return measurements + S_R*np.random.randn(*measurements.shape)


if __name__ == "__main__":
    # scalars that control the environment
    nl = 2 # number of landmarks
    box_size = 100
    num_seconds = 60
    dt = .1
    out_file = 'unicycle_data.npz'
    
    #Allocate everything that will go out
    N = round(num_seconds/dt)
    true_poses = np.zeros(( N+1, 3 ))
    landmark_locs = np.zeros(( nl, 2 ))
    measurements = np.zeros(( N+1, nl ))
    inputs = np.zeros(( N, 2 ))

    # what pattern the unicycle does is going to be a repeating sequence of the
    # `V_command` and `w_command` arrays

    ## Set up velocity commands
    V_one = int(round(5/dt)) #steps to run at 1m/s
    V_two = int(round(3/dt)) # steps to run at 2 m/s
    V_command = np.append(np.ones(V_one),2*np.ones(V_two))
    ## Set up rotation commands
    seconds_straight = 5
    seconds_to_turn=3
    w_straight =int(round(seconds_straight/dt)) # steps to go straight
    w_turn = int(round(seconds_to_turn/dt)) #steps to turn
    ### pi/(2*...) is to make it turn 90 degrees in the turning time. Can modify (obviously)
    w_command = np.append(np.zeros(w_straight), np.ones(w_turn) * pi/(2*seconds_to_turn) )

    ## Now create the inputs array from the V_command and w_command
    for i in range(N):
        inputs[i,0] = V_command[i % len(V_command) ]
        inputs[i,1] = w_command[i % len(V_command) ]

    # Create Landmarks randomly in box, centered at 0,0
    landmark_locs = np.random.rand(nl,2)*box_size - box_size/2


    # How should I corrupt the dynamics?  Define pn_func (process noise function)
    pn_func = partial(process_white_noise, S_Q = np.diag(np.array([.1,.1,.02])*sqrt(dt) ) )
    # How should I corrupt the measurements?  Define mn_func (measurement noise function)
    mn_func = partial(measurement_isotropic_noise, S_R = 1)

    # Generate the truth data and measurements
    
    
    ## Start at 0,0, with a heading of pi/2 ... give or take
    S0 = np.diag(np.array([1., 1., pi/8]))
    initial_error = S0.dot(np.random.randn(3))
    curr_x = process_white_noise(np.array([0.,0., pi/2]), S0)
    true_poses[0]=copy(curr_x)
    measurements[0] = mn_func(noiseless_meas(curr_x, landmark_locs))

    for i in range(N):
        curr_x = pn_func( noiseless_dyn( curr_x, inputs[i], dt) )
        true_poses[i+1] = copy(curr_x)
        measurements[i+1] = mn_func( noiseless_meas( curr_x, landmark_locs ) )
    
    # Done.... store everything out!
    np.savez(out_file, truth=true_poses, landmarks=landmark_locs, inputs=inputs, measurements=measurements)
