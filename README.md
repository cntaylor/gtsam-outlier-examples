# Evaluating Outlier Rejection Techniques in GTSAM
The base of this code is a simple example showing how to do optimization in GTSAM. The code creates a simple unicycle model and estimates where that unicycle is over time using GTSAM.  The measurements at each timestep consist of range measurements to some fixed landmarks.  The core of this file is `unicycle_sim.py`, which runs a simulation that generates truth and the measurement data.  All of that information is stored in a `.npz` file to be read in by an optimization or estimation procedure. 

## Simple Optimization (no faults)
`gtsam_no_outliers.py` is the current optimization code.  It reads in the pertinent information and, assuming there are no outliers, performs the optimization of where the robot was over time.  In addition, this file is an example of how to create your own "custom factor" in GTSAM using the Python bindings.
