import numpy as np
import matplotlib.pyplot as plt

no_outliers = np.load('RMSE_input_no_outlier_est_no_outlier.npy')
meas_10pc_outliers = np.load('RMSE_input_meas_10pc_outlier_est_no_outlier.npy')
no_outliers_huber_est = np.load('RMSE_input_no_outlier_est_huber.npy')
meas_10pc_huber_est = np.load('RMSE_input_meas_10pc_outlier_est_huber.npy')

# Position RMSEs
plt.plot(no_outliers[:,0],label='no outliers')
plt.plot(meas_10pc_outliers[:,0],label='10% outliers, Gaussian Est')
plt.plot(no_outliers_huber_est[:,0],label='no outliers, Huber est')
plt.plot(meas_10pc_huber_est[:,0],label='10% outliers, Huber est')
plt.legend()
plt.title('position RMSEs')
print ("Average position RMSE for no outliers is",np.average(no_outliers[:,0]))
print ("Average position RMSE for 10% outliers is",np.average(meas_10pc_outliers[:,0]))
print ("Average position RMSE for no outliers, Huber is",np.average(no_outliers_huber_est[:,0]))
print ("Average position RMSE for 10% outliers, Huber is",np.average(meas_10pc_huber_est[:,0]))

# Do the angular RMSEs
plt.figure()
plt.plot(no_outliers[:,1],label='no outliers')
plt.plot(meas_10pc_outliers[:,1],label='10% outliers')
plt.plot(no_outliers_huber_est[:,1],label='no outliers, Huber est')
plt.plot(meas_10pc_huber_est[:,1],label='10% outliers, Huber est')
plt.legend()
plt.title('angular RMSEs')
print ("Average angular RMSE for no outliers is",np.average(no_outliers[:,1]))
print ("Average angular RMSE for 10% outliers is",np.average(meas_10pc_outliers[:,1]))
print ("Average position RMSE for no outliers, Huber is",np.average(no_outliers_huber_est[:,1]))
print ("Average position RMSE for 10% outliers, Huber is",np.average(meas_10pc_huber_est[:,1]))

plt.show()