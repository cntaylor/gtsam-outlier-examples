#%%
import numpy as np
import matplotlib.pyplot as plt

# Take output of three files, combine, and graph...
#### This was the original (single file) stuff


file_name1 = 'basic_graph_unicycle_res.npz'
file_name2 = 'switchable_constraints_unicycle_res.npz'
file_name3 = 'discrete_hv_unicycle_res.npz'

data1 = np.load(file_name1)
data2 = np.load(file_name2)
data3 = np.load(file_name3)
# Get some information on what is in the files
in_opts = data1['in_opts']
in_opts2 = data2['in_opts']
in_opts3 = data3['in_opts']
print('in_opts are',in_opts, 'in_opts2 are',in_opts2, 'in_opts3 are',in_opts3)
assert np.all( np.equal(in_opts, in_opts2) ), 'in_opts are not the same, file 1 and 2'
assert np.all( np.equal(in_opts, in_opts3) ), 'in_opts are not the same, file 1 and 3'

n_runs = data1['times'].shape[2]
n_runs2 = data2['times'].shape[2]
n_runs3 = data2['times'].shape[2]
assert n_runs == n_runs2, 'Number of runs different between file 1 and 2'
assert n_runs == n_runs3, 'Number of runs different between file 1 and 3'

# Now to combine stuff together...

## Hack because I names the switchable constraints est-opts badly originally
tmp = data2['est_opts']
est_opts2 = ['SC-'+tmp[i] for i in range(len(tmp))]
## End hack
## Without hack, replace est_opts2 with data2['est_opts']
est_opts = np.concatenate((data1['est_opts'], est_opts2, data3['est_opts']))
pos_RMSEs = np.concatenate((data1['pos_RMSEs'], data2['pos_RMSEs'], data3['pos_RMSEs']), axis=1)
ang_RMSEs = np.concatenate((data1['ang_RMSEs'], data2['ang_RMSEs'], data3['ang_RMSEs']), axis=1)

print ('est_opts are',est_opts)
est_to_plot = np.arange(len(est_opts))
in_to_plot = [0,1,3,5]

fig1, axs = plt.subplots(len(in_to_plot), 1, figsize=(10, len(in_opts) * 5))
for ii, ax in enumerate(axs):
    data_to_plot = [pos_RMSEs[in_to_plot[ii], jj, :] for jj in est_to_plot]
    ax.violinplot(data_to_plot)
    ax.set_xticks(np.arange(1, len(est_to_plot) + 1))
    ax.set_xticklabels(est_opts[est_to_plot])
    ax.set_title(in_opts[in_to_plot[ii]])
    ax.set_xlabel('Estimation Options')
    ax.set_ylabel('Position RMSE (m)')

plt.tight_layout()
plt.savefig('pos_RMSEs.pdf')

fig2, axs = plt.subplots(len(in_to_plot), 1, figsize=(10, len(in_opts) * 5))
for ii, ax in enumerate(axs):
    data_to_plot = [ang_RMSEs[in_to_plot[ii], jj, :] for jj in est_to_plot]
    ax.violinplot(np.array(data_to_plot).T * 180.0/np.pi)
    ax.set_xticks(np.arange(1, len(est_to_plot) + 1))
    ax.set_xticklabels(est_opts[est_to_plot])
    ax.set_title(in_opts[in_to_plot[ii]])
    ax.set_xlabel('Estimation Options')
    ax.set_ylabel('Angular RMSE (degrees)')

plt.tight_layout()
plt.savefig('ang_RMSEs.pdf')

plt.show()
# %%


#### This was the original (single file) stuff

# # file_name = 'switchable_constraints_unicycle_res.npz'
# file_name = 'basic_graph_unicycle_res.npz'

# data = np.load(file_name)
# # Get some information on what is in the file
# in_opts = data['in_opts']
# est_opts = data['est_opts']
# time_res = data['times']
# pos_RMSEs = data['pos_RMSEs']
# ang_RMSEs = data['ang_RMSEs']
# n_runs = data['times'].shape[2]
# est_to_plot = [0,1,2,5,6,7,8,9]
# in_to_plot = [0,1,3,5]

# fig1, axs = plt.subplots(len(in_to_plot), 1, figsize=(10, len(in_opts) * 5))
# for ii, ax in enumerate(axs):
#     data_to_plot = [pos_RMSEs[in_to_plot[ii], jj, :] for jj in est_to_plot]
#     ax.violinplot(data_to_plot)
#     ax.set_xticks(np.arange(1, len(est_to_plot) + 1))
#     ax.set_xticklabels(est_opts[est_to_plot])
#     ax.set_title(in_opts[in_to_plot[ii]])
#     ax.set_xlabel('Estimation Options')
#     ax.set_ylabel('Position RMSE (m)')

# plt.tight_layout()
# plt.savefig('pos_RMSEs.png')

# fig2, axs = plt.subplots(len(in_to_plot), 1, figsize=(10, len(in_opts) * 5))
# for ii, ax in enumerate(axs):
#     data_to_plot = [ang_RMSEs[in_to_plot[ii], jj, :] for jj in est_to_plot]
#     ax.violinplot(np.array(data_to_plot).T * 180.0/np.pi)
#     ax.set_xticks(np.arange(1, len(est_to_plot) + 1))
#     ax.set_xticklabels(est_opts[est_to_plot])
#     ax.set_title(in_opts[in_to_plot[ii]])
#     ax.set_xlabel('Estimation Options')
#     ax.set_ylabel('Angular RMSE (degrees)')

# plt.tight_layout()
# plt.savefig('ang_RMSEs.png')

# fig3, axs = plt.subplots(len(in_to_plot), 1, figsize=(10, len(in_opts) * 5))
# for ii, ax in enumerate(axs):
#     data_to_plot = [time_res[in_to_plot[ii], jj, :] for jj in est_to_plot]
#     ax.violinplot(data_to_plot)
#     ax.set_xticks(np.arange(1, len(est_to_plot) + 1))
#     ax.set_xticklabels(est_opts[est_to_plot])
#     ax.set_title(in_opts[in_to_plot[ii]])
#     ax.set_xlabel('Estimation Options')
#     ax.set_ylabel('Time to Run (s)')

# plt.tight_layout()
# plt.savefig('time_res.png')

# plt.show()
# # %%
