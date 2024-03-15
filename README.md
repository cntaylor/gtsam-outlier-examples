# Evaluating Outlier Rejection Techniques in GTSAM
This project is a collection of scripts that can be used to evaluate the performance of outlier rejection techniques in GTSAM.  The first script of importance is `unicycle_sim.py`.  This script outputs `.npz` files that represent a single run of the simulation.  These `.npz` files can be read in by other scripts that can then use GTSAM (or something else if you want) to optimize for the estimated trajectory of the unicycle. So far, I have one type of graph, which is implemented in `basic_graph.py`.  Within this graph, you can apply different robust estimators, including Huber, Tukey, etc.

# The Simulation script: `unicycle_sim.py`
This script is designed to simulate a unicycle model that generates truth.  It also has some generic functions that can be used to add different types of errors to either the dynamics or the measurements.  Therefore, to change what type of noise is added to either the dynamics or measures, you only have to change the lines that say `pn_func =` for the process noise and `mn_func =` for the measurement noise.  By using different functions, you can create measurements with different percentages of outliers, or with different types of noise, or time correlated noise, or....

This file is setup right now to run the unicycle simulation for a large number of times, storing the results in different `run_*.npz` files.  The directory they are in should hopefully be named so it is obvious what scenario setup the runs are.  For each run, it randomly chooses the location of landmarks.  The "commands" given the unicycle will be the same for each run, but due to process noise the path will vary from run to run.  The measurements are a function of the path and the landmarks (and the measurement noise).

# Estimators (Optimization) files
## `basic_graph.py`
The basic estimator is a gtsam graph (we don't use iSAM2 or such. The examples are not big enough here) which reads in one of the .npz files with the measurements. We assume we know the base covariances (i.e. they are hard-coded / not read in).  We then perform the optimization.  The current code is created so that you can quickly do the optimization for each run in a directory, and apply different modifications to the graph noise models, but the structure of the graph itself is the same.  Note that this file creates a custom factor for a landmark ranging measurement where the landmark is at a known location. This is not part of gtsam by default, but seemed better than adding variable nodes for the landmark locations, and then constraining them to the correct location.

## `switchable_constraints.py`
This estimator runs a GTSAM graph with an extra node for each measurement.  Each node has a prior of '1' (a valid measurement), but this value is used to scale the squared error of each measurement, so it can decrease to 0 to cause error to go down. This means that how "strong" an outlier has to be depends on the "strength" of the prior for the switchable constraint.  This script is setup to run the graph with multiple different variances on those prior constraints

## `discrete_hv_graph.py`
This estimator deviates significantly from traditional GTSAM.  While there is a hybrid factor class in GTSAM, I don't think it was working properly when I used it.  (Maybe I should submit an issue, but I need to come up with a few basic test cases that show why I don't think it works.  Honestly, this is what `simple_discrete.py` is, but it needs to be cleaned up.)

So anyhow, this creates a graph, just like basic_graph, but associated with each measurement is also a discrete variable that can be either True or False for it being an outlier.  If False, it uses the regular measurement covariance, while for True, it uses a much larger covariance.  This graph runs for just a couple of iterations, at which point I evaluate each outlier probability (outside GTSAM) to decide what the boolean variables should be.  A new graph is then formed with the new boolean variables and the iteration repeats.

In all honesty, I think this is just a marginally different approach from the max-mixture approach.  If the hybrid stuff ends up working properly in GTSAM, it could give much different results.  I would like to go this way because I could then start to tie together over time the outliers. So, if a satellite is an outlier at one time step, there is a higher probability it is an outlier at the next step.  That will move it away from max-mixtures, but will require more intelligent processing of the discrete variables than what I am currently doing.  I do believe that what I am doing is what was essentially described in the DCSAM paper though, so it has that going for it!

# Analysis scripts
The different estimators above dump out a bunch of .npz files.  `analyze_results.py` takes the output of a bunch of those and generates plots for accuracy and timing to compare the techniques.

# GNSS
After doing everything with all simulation, also wanted to apply to some real data.  Got some GNSS data with lots of multi-path, or outliers for whatever reason from  
(https://github.com/TUC-ProAut/libRSF)[https://github.com/TUC-ProAut/libRSF].  This is the Protzel dataset used to test switchable constraints and a bunch of other stuff in past papers.

Originally, I tried to just use Python scripts like for the unicycle example.  These files are in `basic_gnss_graph.py`, `switchable_constraints_gnss.py`, and `discrete_hv_gnss_graph.py`.  Unfortunately, while these worked functionally, they were really, really slow.  (Like to solve half the data was taking about 1.5 hours _per iteration_). So, I implemented a pseudorange and clock factor in C++.  This is in the `discrete_normalizing_constants` branch of my forked gtsam repository (https://github.com/cntaylor/gtsam).

The state at each time is a 5 state vector: position(3), clock error, and clock error rate.
For dynamics, I originally tried to only have the clock and clock rate be related over time.  With some robust estimators, and switchable constraints, this led to underconstrained graph, so I also added a very weak equality constraint between timesteps.