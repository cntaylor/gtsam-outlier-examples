# Evaluating Outlier Rejection Techniques in GTSAM

## Simple Optimization (no faults)

The base of this code is a simple example showing how to do optimization in GTSAM. The code creates a simple unicycle model and estimates where that unicycle is over time using GTSAM.  The measurements at each timestep consist of range measurements to some fixed landmarks.  The locations of the landmarks are also estimated.  This works using two files:
* `unicycle_sim.py`
* `simple_opt.py`



The code consists of (`example1-gtsam.py`):

1.  Creating a true path that the unicycle follows, with the associated measurements (up to line 76)
2.  Creating a graph (GTSAM graph) that includes all the measurements and a prior on where the unicycle started and where the landmark locations are
3.  Optimizing the graph
4.  Showing the output

The other file, examples.py, includes some helper functions to make this all work.

