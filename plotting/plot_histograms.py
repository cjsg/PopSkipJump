import pylab as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatterMathtext


BETA = 1.0
data = np.load(f"adv/sigmoids/sigmoids_v2_{BETA}_0_499.npy")

n_bins = 50
n_images = data.shape[1]
cnt = 1
plt.figure(figsize=(9, 35))
plt.suptitle(f'BETA: {BETA}')
for i in list(range(0, 4)) + list(range(4, 32, 5)):
    s = data[i, :, 2]
    t = data[i, :, 3]
    s_bins = np.logspace(0, 2.1, n_bins)
    t_bins = np.linspace(-1, 1, n_bins)
    s_indices = np.digitize(s, s_bins)
    t_indices = np.digitize(t, t_bins)
    s_freq = np.array([np.sum((s_indices == i) * 1) for i in range(n_bins)]) / n_images
    t_freq = np.array([np.sum((t_indices == i) * 1) for i in range(n_bins)]) / n_images
    s_t_freq = np.matmul(np.expand_dims(t_freq, axis=1), np.expand_dims(s_freq, axis=0))

    ax = plt.subplot(10, 2, cnt)
    ax.set_xlabel('s')
    ax.set_ylabel('t')
    plt.title(f'Iteration {i}: p(s,t) = p(s)*p(t)')
    X, Y = np.meshgrid(s_bins, t_bins)
    im = plt.pcolor(X, Y, s_t_freq, cmap='RdPu')
    plt.xscale('log')
    plt.colorbar(im, orientation='horizontal', format=LogFormatterMathtext())

    st_freq = np.zeros(shape=s_t_freq.shape)
    for j in range(n_images):
        st_freq[t_indices[j], s_indices[j]] += 1
    st_freq /= n_images

    ax = plt.subplot(10, 2, cnt+1)
    im = plt.pcolor(X, Y, st_freq, cmap='RdPu')
    plt.xscale('log')
    plt.colorbar(im, orientation='horizontal', format=LogFormatterMathtext())
    ax.set_xlabel('s')
    ax.set_ylabel('t')
    plt.title(f'Iteration {i}: obtained p(s,t)')
    cnt+=2

plt.savefig(f"adv/sigmoids/joint_st_v2_{BETA}.pdf")