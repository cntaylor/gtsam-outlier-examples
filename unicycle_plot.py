import numpy as np
import matplotlib.pyplot as plt

def plot_locs(state_array):
    '''
    Take in a Nx3 array and plot the locations over time
    '''
    plt.plot(state_array[:,0], state_array[:,1])
    # TODO  add a start and stop marker, with a legend

if __name__ == '__main__':
    data = np.load('unicycle_data.npz')
    plot_locs(data['truth'])
    plt.show()