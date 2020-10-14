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


def estimate_repeat_in_hsj(beta, repeats):
    repeats = [int(x) for x in repeats.split()]
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


def psj_vs_hsjr(R):
    noise = 'bayesian'
    flip = "0.00"
    n_samples = "100"
    ratios = []
    betas = list(R.keys())
    for beta in betas:
        psj_exp_name = f"psj_r_1_b_{beta}_{noise}_fp_{flip}_ns_{n_samples}"
        psj_dump = read_dump(psj_exp_name)
        psj_calls = np.median(psj_dump['model_calls'][-1])
        hsj_exp_name = f"hsj_rep_r_{R[beta]}_b_{beta}_{noise}_fp_{flip}_ns_{n_samples}"
        hsj_dump = read_dump(hsj_exp_name)
        hsj_calls = np.median(hsj_dump['model_calls'][-1])
        ratio = hsj_calls / psj_calls
        ratios.append(ratio)
    plt.figure(figsize=(7, 7))
    image_path = f'{PLOTS_DIR}/exp3'
    plt.plot(betas, ratios)
    plt.plot()
    plt.xlabel('beta')
    plt.ylabel('Model Calls (in .x)')
    plt.legend()
    # plt.yscale('log')
    plt.grid()
    plt.savefig(f'{image_path}.png', bbox_inches='tight', pad_inches=.02)
    plt.savefig(f'{image_path}.pdf', bbox_inches='tight', pad_inches=.02)


def conv_to_hsja():
    noise = 'bayesian'
    flip = "0.00"
    n_samples = "100"
    betas = [1, 1.25, 1.5, 2, 5]
    attacks = ['psj', 'hsj']
    labels = [f'PSJ: beta={b}' for b in betas] + [f'{a.upper()}: Det' for a in attacks]
    dist_arr, calls_arr = [], []
    for beta in betas:
        psj_exp_name = f"psj_r_1_b_{beta}_{noise}_fp_{flip}_ns_{n_samples}"
        psj_dump = read_dump(psj_exp_name)
        psj_calls = np.median(psj_dump['model_calls'], axis=1)
        psj_dist = np.median(psj_dump['border_distance'], axis=1)
        dist_arr.append(psj_dist)
        calls_arr.append(psj_calls)

    for attack in attacks:
        exp_name = f"{attack}_r_1_b_1_deterministic_fp_{flip}_ns_{n_samples}"
        exp_dump = read_dump(exp_name)
        exp_calls = np.median(exp_dump['model_calls'], axis=1)
        exp_dist = np.median(exp_dump['border_distance'], axis=1)
        dist_arr.append(exp_dist)
        calls_arr.append(exp_calls)

    plt.figure(figsize=(12, 7))
    image_path = f'{PLOTS_DIR}/exp2_dist_vs_calls'
    for i in range(len(labels)):
        plt.plot(calls_arr[i], dist_arr[i], label=f"{labels[i]}")
    plt.legend()
    plt.xscale('log')
    plt.xlabel('Median Model Calls')
    plt.ylabel('Median Border Distance')
    plt.grid()
    plt.savefig(f'{image_path}.png', bbox_inches='tight', pad_inches=.02)
    plt.savefig(f'{image_path}.pdf', bbox_inches='tight', pad_inches=.02)

    plt.figure(figsize=(12, 7))
    image_path = f'{PLOTS_DIR}/exp2_dist_vs_rounds'
    for i in range(len(labels)):
        plt.plot(dist_arr[i], label=f"{labels[i]}")
    plt.legend()
    plt.xlabel('Rounds')
    plt.ylabel('Median Border Distance')
    plt.grid()
    plt.savefig(f'{image_path}.png', bbox_inches='tight', pad_inches=.02)
    plt.savefig(f'{image_path}.pdf', bbox_inches='tight', pad_inches=.02)


def hsj_failure():
    beta = "1"
    noise = "stochastic"
    n_samples = "100"
    attacks = ['hsj', 'hsj_rep', 'psj']
    repeats = [1, 3, 1]
    flips = ['0.00', '0.05', '0.10', '0.15']
    print('****\t'+'\t'.join(attacks))
    for flip in flips:
        print(flip, end='\t')
        for repeat, attack in zip(repeats, attacks):
            exp_name = f"{attack}_r_{repeat}_b_{beta}_{noise}_fp_{flip}_ns_{n_samples}"
            raw = read_dump(exp_name)
            calls = np.median(raw['model_calls'][-1])
            dist = np.median(raw['border_distance'][-1])
            P_OUT = raw['prob_true_label_out'].cpu().numpy() * 1
            is_adv = (P_OUT[-1, :] < 0.5) * 1
            false_prop = 1 - np.sum(is_adv) / float(n_samples)
            # print(f'({flip},{attack})\t', end='')
            print(false_prop.round(2), end=', ')
            # print(false_prop.round(2), dist, calls)
        print('')


# estimate_repeat_in_hsj(beta=10)
beta_vs_repeats = {
    1: "3072000 4096000 6144000 8192000",
    2: "256000 384000 512000 640000 1024000 2048000 3072000 4096000 6144000 8192000",
    5: "16000 32000 64000 128000 256000 384000 512000 640000 1024000 2048000",
    10: "200 500 1000 2000 4000 8000 16000 32000 64000 128000 192000 256000",
    20: "2000 4000 8000 16000 32000 64000 72000 128000",
    50: "2000 4000 8000 16000",
    100: "2000 4000 8000 16000",
    200: "50 100 200 500 850 1000 2000 4000 8000 16000",
}

best_repeat = {
    2: "8192000",
    5: "1024000",
    10: "192000",
    20: "72000",
    50: "16000",
    100: "9000",
    200: "850",
}
rc('text', usetex=True)
rc('text.latex', preamble=[r'\usepackage{amsfonts}'])
rc('font', size=14)  # default: 10 -> choose size depending on figsize
rc('font', family='STIXGeneral')
rc('legend', fontsize=16)
plt.tight_layout(h_pad=0, w_pad=.5)


# for beta in beta_vs_repeats:
#     estimate_repeat_in_hsj(beta=beta, repeats=beta_vs_repeats[beta])
hsj_failure()
# psj_vs_hsjr(best_repeat)
# conv_to_hsja()
# plot_distance()