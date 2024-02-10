#%%
import numpy as np
import matplotlib.pyplot as plt

# file_name = 'switchable_constraints_unicycle_res.npz'
file_name = 'basic_graph_unicycle_res.npz'

data = np.load(file_name)
# Get some information on what is in the file
in_opts = data['in_opts']
est_opts = data['est_opts']
time_res = data['times']
pos_RMSEs = data['pos_RMSEs']
ang_RMSEs = data['ang_RMSEs']
n_runs = data['times'].shape[2]
est_to_plot = [0,1,2,5,6,7,8,9]
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
plt.savefig('pos_RMSEs.png')

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
plt.savefig('ang_RMSEs.png')

fig3, axs = plt.subplots(len(in_to_plot), 1, figsize=(10, len(in_opts) * 5))
for ii, ax in enumerate(axs):
    data_to_plot = [time_res[in_to_plot[ii], jj, :] for jj in est_to_plot]
    ax.violinplot(data_to_plot)
    ax.set_xticks(np.arange(1, len(est_to_plot) + 1))
    ax.set_xticklabels(est_opts[est_to_plot])
    ax.set_title(in_opts[in_to_plot[ii]])
    ax.set_xlabel('Estimation Options')
    ax.set_ylabel('Time to Run (s)')

plt.tight_layout()
plt.savefig('time_res.png')

plt.show()
# %%
