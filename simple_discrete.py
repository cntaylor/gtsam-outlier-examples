#%%
'''
The goal of this file is to provide a simple test of how discrete
factors might work with gtsam.  Primarily, there will be a single
hidden variable, a double value, that will have a single prior.  
It will also have several measurements, and associated with
each measurement will be a discrete factor evaluating the probability
that the measurement is valid.  Some measurements will be corrupted
with white noise, and some with outlier (much larger white) noise.
These inputs will then be put into gtsam and we will evaluate how 
accurately the discrete variables estimate which ones were outliers.
'''
import numpy as np
import gtsam
import matplotlib.pyplot as plt
from typing import Optional
import math as m

# function for generating data measurements and which ones 
# are outliers

def create_measurements(truth : float,
                        N : int, 
                        outlier_prob : float = .1, 
                        outlier_mult : float = 15.,
                        R : float = 1.) -> tuple:
    '''
    This function creates a set of measurements and which ones are outliers
    Inputs:
        truth : the true value which noiseless measurements would return
        N : the number of measurements to create
        outlier_prob : the probability of each measurement to be corrupted by an outlier (should be 0 < % < 1, checked)
        outlier_mult : What to multiply S_R by to get the outlier covariance (should be >1, checked)
        R : the variance value used to do non-outlier noise
    Outputs:
        A tuple of (meas, outlier_bools), where outlier_bools is an 
        array of booleans indicating which measurements are outliers (True=it is an outlier)
    '''
    assert 0 < outlier_prob < 1, "outlier_prob should be 0 < % < 1"
    assert outlier_mult > 1, "outlier_mult should be > 1"

    meas = truth + m.sqrt(R)*np.random.randn(N)
    outlier_bools = np.random.rand(N) < outlier_prob
    n_outliers = np.sum(outlier_bools)
    meas[outlier_bools] = truth + outlier_mult* m.sqrt(R) * np.random.randn(n_outliers)
    return meas, outlier_bools


### This is bad coding practice, but this is just a test, so...
# define some symbols here for use throughout the rest of the file
prior_var = 16. # variance on the prior factor
main_hv = gtsam.symbol('x',0) # main hidden variable
# outlier hidden variable 'key' 
outlier_hv = lambda i : gtsam.symbol('o',i)

def create_graph(Z : np.array, prior_val: float,
                 R : float = 1.,
                 outlier_mult : float = 1000.,
                 prior_prob_outlier : Optional[float] = None) \
            -> gtsam.HybridNonlinearFactorGraph:
    '''
    This function creates a gtsam graph with a single continuous hidden
    variable, a prior factor for that variable, and a bunch of measurements,
    with a discrete factor for each measurement
    '''

    inlier_noise = gtsam.noiseModel.Diagonal.Variances(np.array([R]))
    outlier_noise = gtsam.noiseModel.Diagonal.Variances(np.array([R*outlier_mult**2.]))
    prior_noise = gtsam.noiseModel.Diagonal.Variances(np.array([prior_var]))
    
    graph = gtsam.HybridNonlinearFactorGraph()
    graph.push_back(gtsam.PriorFactorDouble(main_hv, prior_val, prior_noise) )
    for i in range(1):
        # Python bindings for gtsam and DiscreteKeys are not great yet (Feb 2024)
        # Need to create a DiscreteKeys object here
        dk = gtsam.DiscreteKeys()
        # And then add each key to it, just to pass it into the MixtureFactor
        dk.push_back( (outlier_hv(i), 2) )
        # This command adds the actual factor
        graph.push_back(gtsam.MixtureFactor([main_hv], dk,
                                      [gtsam.PriorFactorDouble(main_hv, Z[i], inlier_noise),
                                       gtsam.PriorFactorDouble(main_hv, Z[i], outlier_noise) 
                                      ]))
    return graph

#%%
if __name__ == '__main__':
    # scalars that control the environment
    N = 10 # number of measurements to create
    outlier_prob = 0.000000001 # probability of each measurement to be corrupted by an outlier
    outlier_mult = 15. # What to multiply S_R by to get the outlier covariance
    R = prior_var #1. # variance value used to do non-outlier noise

    # First, setup the prior values
    truth = 5.
    prior_val = truth + np.random.randn() * m.sqrt(prior_var)
    Z, outlier_truth = create_measurements(truth, N, outlier_prob, outlier_mult, R)
    Z[0] = prior_val
    graph = create_graph(Z, prior_val, R, outlier_mult=1.1)
    values = gtsam.Values()
    values.insert(main_hv, prior_val)
    # for i in range(N):
    #     values.insert(outlier_hv(i), 0)
    # Now, evaluate the graph
    curr_lin_graph = graph.linearize(values)
    forward_pass = curr_lin_graph.eliminateSequential()
    res_values = forward_pass.optimize()

    # So, the problem is res_values returns the solution to the
    # _linearized_ graph.  In other words, it returns the Delta
    # to get you closer to the solution, but not the solution itself.

    # Now, plot the results
    print('The true value is', truth, 'the final estimate was', res_values.at(main_hv) + values.atDouble(main_hv))

    # Plotting the bar chart for outlier_truth vs backward_pass values
    outlier_estimates = [res_values.atDiscrete(outlier_hv(i)) for i in range(N)]
    indices = np.arange(N)
    bar_width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar(indices, outlier_truth, bar_width, label='True Outliers')
    plt.bar(indices + bar_width, outlier_estimates, bar_width, label='Estimated Outliers')

    plt.xlabel('Measurement Index')
    plt.ylabel('Outlier Value')
    plt.title('Comparison of True Outliers and Estimated Outliers')
    plt.xticks(indices + bar_width / 2, indices)
    plt.legend()

    plt.tight_layout()
    plt.show()

# %%
