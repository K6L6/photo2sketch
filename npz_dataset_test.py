import numpy as np
import ipdb

from svg_parser import view_stroke3
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def plot_many_sketches(sketches, n_row=10, n_col=10):
    fig = plt.figure(figsize=(7, 7))

    n = 0
    gs = gridspec.GridSpec(n_row, n_col, wspace=0.1, hspace=0.1)
    for i in range(n_row):
        for j in range(n_row):
            ax = fig.add_subplot(gs[i, j])
            view_stroke3(sketches[n], ax)
            ax.set_xticks([])
            ax.set_yticks([])
            n += 1
    fig.tight_layout()
    
    return fig

def run_tests():
    """ run tests using npz dataset. """
    google_dataset = 'data/aaron_sheep.npz'
    my_dataset = 'data/airplane.npz'
    
    def get_min_max(data):
        vmin, vmax = 0, 0
        for d in data:
            vmin = min(vmin, d.min())
            vmax = max(vmax, d.max())
        return vmin, vmax
    
    # vmin = -476, vmax = 460
    google_dataset = np.load(google_dataset, encoding='latin1')['train']
    vmin, vmax = get_min_max(google_dataset)
    print('google: min = ', vmin, ', max = ', vmax)

    fig = plot_many_sketches(google_dataset)
    fig.suptitle('aaron sheep')

    my_dataset = np.load(my_dataset)['train']
    vmin, vmax = get_min_max(my_dataset)
    print('my_dataset: min = ', vmin, ', max = ', vmax)

    fig = plot_many_sketches(my_dataset)
    fig.suptitle('airplane (my dataset)')
    
    plt.show()

if __name__ == '__main__':
    run_tests()