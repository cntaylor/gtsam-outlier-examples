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

fig1, axs = plt.subplots(len(in_opts), 1, figsize=(10, len(in_opts) * 5))
for ii, ax in enumerate(axs):
    data_to_plot = [pos_RMSEs[ii, jj, :] for jj in range(len(est_opts))]
    ax.violinplot(data_to_plot)
    ax.set_xticks(np.arange(1, len(est_opts) + 1))
    ax.set_xticklabels(est_opts[:, 0])
    ax.set_title(in_opts[ii, 0])
    ax.set_xlabel('Estimation Options')
    ax.set_ylabel('Position RMSE (m)')

plt.tight_layout()

fig2, axs = plt.subplots(len(in_opts), 1, figsize=(10, len(in_opts) * 5))
for ii, ax in enumerate(axs):
    data_to_plot = [ang_RMSEs[ii, jj, :] for jj in range(len(est_opts))]
    ax.violinplot(data_to_plot*180/np.pi)
    ax.set_xticks(np.arange(1, len(est_opts) + 1))
    ax.set_xticklabels(est_opts[:, 0])
    ax.set_title(in_opts[ii, 0])
    ax.set_xlabel('Estimation Options')
    ax.set_ylabel('Angular RMSE (degrees)')

plt.tight_layout()

fig3, axs = plt.subplots(len(in_opts), 1, figsize=(10, len(in_opts) * 5))
for ii, ax in enumerate(axs):
    data_to_plot = [time_res[ii, jj, :] for jj in range(len(est_opts))]
    ax.violinplot(data_to_plot)
    ax.set_xticks(np.arange(1, len(est_opts) + 1))
    ax.set_xticklabels(est_opts[:, 0])
    ax.set_title(in_opts[ii, 0])
    ax.set_xlabel('Estimation Options')
    ax.set_ylabel('Time to Run (s)')

plt.tight_layout()

plt.show()
# %%
