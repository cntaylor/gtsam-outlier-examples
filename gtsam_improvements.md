Keeping a list of things that might be worth doing in the future to contribute to GTSAM:

* Example of Discrete networks in Python
* Maybe a simple description of what things mean / how to use
* adding ".add" to HybridGaussianFactorGraph (just an alias for push_back, but works for other networks)
* priorFactorDouble allow 3rd parameter to be a Double (rather than a complete noiseModel)
* expose DiscreteKey to Python
* expose automatic conversion to DiscreteKeys from a DiscreteKey or list of 
* Should HybridNonlinearFactorGraph allow for a priorFactorDouble?
* Make return of optimize give full, not delta values
* Not positive yet, but I believe they are not taking into account the normalization constant for the Gaussians in the MixtureFactor.  (If things are at the same mean, it will always peak the thing with the larger covariance because its cost is always lower without the noramlization constant)