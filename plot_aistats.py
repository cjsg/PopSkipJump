import os

import matplotlib.pylab as plt
import numpy as np
import torch
from matplotlib import rc

from img_utils import get_device

OUT_DIR = 'aistats'
PLOTS_DIR = f'{OUT_DIR}/plots_aistats/'
device = get_device()
NUM_ITERATIONS = 32
NUM_IMAGES = 1
eps = list(range(1, 6))


def read_dump(path):
    filepath = f'{OUT_DIR}/{path}/crunched.pkl'
    raw = torch.load(open(filepath, 'rb'), map_location=device)
    return raw


def estimate_repeat_in_hsj(beta=10):
    repeats = "200 500 1000 2000 4000 8000 16000 32000 64000 128000".split()
    repeats = [int(x) for x in repeats]
    attack = 'hsj_rep'
    noise = 'bayesian'
    flip = "0.00"
    n_samples = "100"
    dists = list()
    for repeat in repeats:
        exp_name = f"{attack}_r_{repeat}_b_{beta}_{noise}_fp_{flip}_ns_{n_samples}"
        raw = read_dump(exp_name)
        D = raw['border_distance']
        dists.append(np.median(D[-1]))
    dists = np.array(dists)
    raw = read_dump(f"psj_r_1_b_{beta}_{noise}_fp_{flip}_ns_{n_samples}")
    D = raw['border_distance']
    medians = np.array([np.median(D[-1])] * len(repeats))
    perc40 = np.array([np.percentile(D[-1], 40)] * len(repeats))
    perc60 = np.array([np.percentile(D[-1], 60)] * len(repeats))
    plt.figure(figsize=(10, 7))
    image_path = f'{PLOTS_DIR}/repeat_beta_{beta}'
    plt.plot(repeats, dists, label='HSJ-R')
    plt.plot(repeats, medians, label='PSJ')
    plt.fill_between(repeats, perc40, perc60, alpha=0.2)
    plt.plot()
    plt.legend()
    plt.xscale('log')
    plt.grid()
    plt.savefig(f'{image_path}.png', bbox_inches='tight', pad_inches=.02)
    plt.savefig(f'{image_path}.pdf', bbox_inches='tight', pad_inches=.02)


def plot_distance():
    beta = "10"
    noise = "bayesian"
    flip = "0.00"
    n_samples = "100"
    # raw = read_dump(f"hsj_rep_r_257_b_{beta}_{noise}_fp_{flip}_ns_{n_samples}")
    raw = read_dump("del_later")
    D = raw['border_distance']
    plt.figure(figsize=(10, 7))
    image_path = f'{PLOTS_DIR}/dist'
    plt.plot(np.median(D, axis=1), label='PSJ')
    plt.plot()
    plt.legend()
    plt.grid()
    plt.savefig(f'{image_path}.png', bbox_inches='tight', pad_inches=.02)
    plt.savefig(f'{image_path}.pdf', bbox_inches='tight', pad_inches=.02)


estimate_repeat_in_hsj(beta=10)
# plot_distance()