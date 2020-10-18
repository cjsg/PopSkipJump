import os

import matplotlib.pylab as plt
import numpy as np
import torch
from matplotlib import rc

from img_utils import get_device
from exp3_constants import beta_vs_repeats, best_repeat

OUT_DIR = 'aistats'
PLOTS_DIR = f'{OUT_DIR}/plots_aistats/'
device = get_device()
NUM_ITERATIONS = 32
NUM_IMAGES = 100
eps = list(range(1, 6))


def read_dump(path):
    filepath = f'{OUT_DIR}/{path}/crunched.pkl'
    raw = torch.load(open(filepath, 'rb'), map_location=device)
    return raw


def estimate_repeat_in_hsj(beta_vs_repeats, dataset):
    attacks = ['hsj_rep', 'hsj_rep_psj_delta']
    labels = ['HSJ-R', 'HSJ-Theta']
    noise = 'bayesian'
    flip = "0.00"
    n_samples = "100"
    for beta in [2, 5, 10, 20, 50, 100, 200]:
        plt.figure(figsize=(10, 7))
        image_path = f'{PLOTS_DIR}/repeat_beta_{beta}'
        min_tick, max_tick = 10**9, 0
        for j, attack in enumerate(attacks):
            if beta not in beta_vs_repeats[attack][dataset]:
                continue
            repeats = beta_vs_repeats[attack][dataset][beta]
            repeats = [int(x) for x in repeats.split()]
            min_tick, max_tick = min(min_tick, repeats[0]), max(max_tick, repeats[-1])
            dists, dists_low, dists_high = list(), list(), list()
            for repeat in repeats:
                exp_name = f"{attack}_r_{repeat}_b_{beta}_{noise}_fp_{flip}_ns_{n_samples}"
                raw = read_dump(exp_name)
                D = raw['border_distance']
                dists.append(np.median(D[-1]))
                dists_low.append(np.percentile(D[-1], 40))
                dists_high.append(np.percentile(D[-1], 60))
            plt.plot(repeats, dists, label=labels[j])
            plt.fill_between(repeats, dists_low, dists_high, alpha=0.2)
        raw = read_dump(f"psj_r_1_b_{beta}_{noise}_fp_{flip}_ns_{n_samples}")
        D = raw['border_distance']
        ticks = np.logspace(np.log10(min_tick), np.log10(max_tick))
        medians = np.array([np.median(D[-1])] * len(ticks))
        perc40 = np.array([np.percentile(D[-1], 35)] * len(ticks))
        perc60 = np.array([np.percentile(D[-1], 65)] * len(ticks))
        plt.plot(ticks, medians, label='PSJ')
        plt.fill_between(ticks, perc40, perc60, alpha=0.2)
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
    ratios, ratios_40, ratios_60 = [], [], []
    betas = list(R.keys())
    for beta in betas:
        psj_exp_name = f"psj_r_1_b_{beta}_{noise}_fp_{flip}_ns_{n_samples}"
        psj_dump = read_dump(psj_exp_name)
        psj_calls = np.median(psj_dump['model_calls'][-1])
        psj_calls_40 = np.percentile(psj_dump['model_calls'][-1], 30)
        psj_calls_60 = np.percentile(psj_dump['model_calls'][-1], 70)

        hsj_exp_name = f"hsj_rep_r_{R[beta][1]}_b_{beta}_{noise}_fp_{flip}_ns_{n_samples}"
        hsj_dump = read_dump(hsj_exp_name)
        hsj_calls = np.median(hsj_dump['model_calls'][-1])

        hsj_l = read_dump(f"hsj_rep_r_{R[beta][0]}_b_{beta}_{noise}_fp_{flip}_ns_{n_samples}")
        hsj_calls_l = np.median(hsj_l['model_calls'][-1])
        hsj_u = read_dump(f"hsj_rep_r_{R[beta][2]}_b_{beta}_{noise}_fp_{flip}_ns_{n_samples}")
        hsj_calls_u = np.median(hsj_u['model_calls'][-1])

        ratio = hsj_calls / psj_calls
        ratios.append(ratio)
        ratios_40.append(hsj_calls_l / psj_calls)
        ratios_60.append(hsj_calls_u / psj_calls)
    plt.figure(figsize=(7, 5))
    image_path = f'{PLOTS_DIR}/exp3'
    plt.plot(betas, ratios)
    plt.fill_between(betas, ratios_40, ratios_60, alpha=0.2)
    plt.plot()
    plt.xlabel('inverse temperature')
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.ylabel('relative number of model calls')
    # plt.legend()
    plt.yscale('log')
    plt.grid()
    plt.savefig(f'{image_path}.png', bbox_inches='tight', pad_inches=.02)
    plt.savefig(f'{image_path}.pdf', bbox_inches='tight', pad_inches=.02)


def conv_to_hsja():
    noise = 'bayesian'
    flip = "0.00"
    n_samples = "50"
    betas = [1, 1.25, 1.5, 2]
    attacks = ['psj', 'hsj']
    # stri = '$\\frac{1}{T}$'
    labels = ['PSJ: $T$=%.2f' % (1/b) for b in betas] + [f'{a.upper()}: $T$=0 (det)' for a in attacks]
    datasets = ['mnist', 'cifar10']
    for dataset in datasets[-1:]:
        dist_arr, calls_arr = [], []
        for beta in betas:
            if dataset == 'mnist':
                psj_exp_name = f"psj_r_1_b_{beta}_{noise}_fp_{flip}_ns_{n_samples}"
            else:
                psj_exp_name = f"{dataset}_psj_r_1_b_{beta}_{noise}_fp_{flip}_ns_{n_samples}"
            psj_dump = read_dump(psj_exp_name)
            psj_calls = np.median(psj_dump['model_calls'], axis=1)
            psj_dist = np.median(psj_dump['border_distance'], axis=1)
            dist_arr.append(psj_dist)
            calls_arr.append(psj_calls)

        for attack in attacks:
            if dataset == 'mnist':
                exp_name = f"{attack}_r_1_b_1_deterministic_fp_{flip}_ns_{n_samples}"
            else:
                exp_name = f"{dataset}_{attack}_r_1_b_1_deterministic_fp_{flip}_ns_{n_samples}"
            exp_dump = read_dump(exp_name)
            exp_calls = np.median(exp_dump['model_calls'], axis=1)
            exp_dist = np.median(exp_dump['border_distance'], axis=1)
            dist_arr.append(exp_dist)
            calls_arr.append(exp_calls)

        plt.figure(figsize=(12, 7))
        image_path = f'{PLOTS_DIR}/exp2_dist_vs_calls_{dataset}'
        for i in range(len(labels)):
            plt.plot(calls_arr[i], dist_arr[i], label=f"{labels[i]}")
        plt.legend()
        plt.xscale('log')
        plt.xlabel('median model calls')
        plt.xlim(10)
        plt.ylabel('median border distance')
        plt.grid()
        plt.savefig(f'{image_path}.png', bbox_inches='tight', pad_inches=.02)
        plt.savefig(f'{image_path}.pdf', bbox_inches='tight', pad_inches=.02)

        plt.figure(figsize=(12, 7))
        image_path = f'{PLOTS_DIR}/exp2_dist_vs_rounds_{dataset}'
        for i in range(len(labels)):
            plt.plot(dist_arr[i], label=f"{labels[i]}")
        plt.legend()
        plt.xlabel('iterations')
        plt.ylabel('median border distance')
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
    datasets = ['mnist', 'cifar10']
    for dataset in datasets:
        print(f'=========={dataset}=========')
        print('****\t'+'\t'.join(attacks))
        for flip in flips:
            print(flip, end='\t')
            for repeat, attack in zip(repeats, attacks):
                if dataset == 'mnist':
                    exp_name = f"{attack}_r_{repeat}_b_{beta}_{noise}_fp_{flip}_ns_{n_samples}"
                else:
                    exp_name = f"{dataset}_{attack}_r_{repeat}_b_{beta}_{noise}_fp_{flip}_ns_{n_samples}"
                raw = read_dump(exp_name)
                calls = np.median(raw['model_calls'][-1])
                dist = np.median(raw['attack_out_distance'][-1])
                # print(f'({flip},{attack})\t', end='')
                print(f'({calls}, {dist})', end='\t')
            print('')

def plot_distance():
    plt.figure(figsize=(10, 7))
    exp_name = f"cifar10_psj_r_1_b_1_stochastic_fp_0.10_ns_5"
    raw = read_dump(exp_name)
    calls = np.median(raw['model_calls'][-1])
    dist = np.median(raw['border_distance'], axis=1)
    plt.plot(dist, label='PSJ')
    plt.plot()
    plt.legend()
    # plt.xscale('log')
    plt.grid()
    plt.show()

# estimate_repeat_in_hsj(beta=10)

rc('text', usetex=True)
rc('text.latex', preamble=[r'\usepackage{amsfonts}'])
rc('font', size=14)  # default: 10 -> choose size depending on figsize
rc('font', family='STIXGeneral')
rc('legend', fontsize=16)
plt.tight_layout(h_pad=0, w_pad=.5)

attack='hsj_rep_psj_delta'
dataset='mnist'
estimate_repeat_in_hsj(beta_vs_repeats, dataset)
# hsj_failure()
# psj_vs_hsjr(best_repeat[attack][dataset])
# conv_to_hsja()
# plot_distance()