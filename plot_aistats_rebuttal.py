import os

import matplotlib.pylab as plt
import numpy as np
import torch
from tqdm import tqdm
from matplotlib import rc

from img_utils import get_device
from model_factory import get_model
from exp3_constants import beta_vs_repeats, best_repeat

OUT_DIR = 'aistats'
PLOTS_DIR = f'{OUT_DIR}/plots_aistats/'
device = get_device()
NUM_ITERATIONS = 32
NUM_IMAGES = 100
eps = list(range(1, 6))


def read_dump(path, raw=False):
    if raw:
        filepath = f'{OUT_DIR}/{path}/raw_data.pkl'
    else:
        filepath = f'{OUT_DIR}/{path}/crunched_new.pkl'
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
        min_tick, max_tick = 10 ** 9, 0
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


def psj_vs_hsjr(best_repeat):
    noise = 'bayesian'
    flip = "0.00"
    n_samples = "100"
    plt.figure(figsize=(7, 5))
    image_path = f'{PLOTS_DIR}/exp3'
    for attack in best_repeat:
        R = best_repeat[attack]['mnist']
        ratios, ratios_40, ratios_60 = [], [], []
        betas = list(R.keys())
        for beta in betas:
            psj_exp_name = f"psj_r_1_b_{beta}_{noise}_fp_{flip}_ns_{n_samples}"
            psj_dump = read_dump(psj_exp_name)
            psj_calls = np.median(psj_dump['model_calls'][-1])
            psj_calls_40 = np.percentile(psj_dump['model_calls'][-1], 30)
            psj_calls_60 = np.percentile(psj_dump['model_calls'][-1], 70)

            hsj_exp_name = f"{attack}_r_{R[beta][1]}_b_{beta}_{noise}_fp_{flip}_ns_{n_samples}"
            hsj_dump = read_dump(hsj_exp_name)
            hsj_calls = np.median(hsj_dump['model_calls'][-1])

            hsj_l = read_dump(f"{attack}_r_{R[beta][0]}_b_{beta}_{noise}_fp_{flip}_ns_{n_samples}")
            hsj_calls_l = np.median(hsj_l['model_calls'][-1])
            hsj_u = read_dump(f"{attack}_r_{R[beta][2]}_b_{beta}_{noise}_fp_{flip}_ns_{n_samples}")
            hsj_calls_u = np.median(hsj_u['model_calls'][-1])

            ratio = hsj_calls / psj_calls
            ratios.append(ratio)
            ratios_40.append(hsj_calls_l / psj_calls)
            ratios_60.append(hsj_calls_u / psj_calls)

        plt.plot(betas, ratios)
        plt.fill_between(betas, ratios_40, ratios_60, alpha=0.2)
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
    n_samples = "20"
    betas = [0.5, 1]
    # betas = [1, 1.5, 2, 10]
    attacks = ['psj', 'hsj', 'hsj_rep_psj_delta']
    # stri = '$\\frac{1}{T}$'
    labels = ['PSJ: T=%.2f' % (1 / b) for b in betas] + [f'{a.upper()}: T=0 (det)' for a in attacks]
    labels = ['PSJ: T=%.2f' % (1 / b) for b in betas] + ['PSJ T=0 (det)', 'HSJ T=0 (det)', 'HSJ-D T=0 (det)']
    # labels = ['PSJ', 'HSJ',  'HSJ-D']
    datasets = ['mnist', 'cifar10']
    for dataset in datasets[-1:]:
        dist_arr, calls_arr = [], []
        for beta in betas:
            if False and dataset == 'mnist':
                psj_exp_name = f"psj_r_1_b_{beta}_{noise}_fp_{flip}_ns_{n_samples}"
            else:
                psj_exp_name = f"{dataset}_psj_r_1_b_{beta}_{noise}_fp_{flip}_ns_{n_samples}"
            psj_dump = read_dump(psj_exp_name)
            psj_calls = np.median(psj_dump['model_calls'], axis=1)
            psj_dist = np.median(psj_dump['border_distance'], axis=1)
            dist_arr.append(psj_dist)
            calls_arr.append(psj_calls)

        for i, attack in enumerate(attacks):
            if False and dataset == 'mnist':
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
        # plt.savefig(f'{image_path}.png', bbox_inches='tight', pad_inches=.02)
        plt.savefig(f'{image_path}.pdf', bbox_inches='tight', pad_inches=.02)

        plt.figure(figsize=(12, 7))
        image_path = f'{PLOTS_DIR}/calls_vs_rounds_{dataset}'
        for i in range(len(labels)):
            plt.plot(calls_arr[i], label=f"{labels[i]}")
        plt.legend()
        plt.xlabel('iterations')
        plt.ylabel('model calls')
        plt.grid()
        # plt.savefig(f'{image_path}.png', bbox_inches='tight', pad_inches=.02)
        plt.savefig(f'{image_path}.pdf', bbox_inches='tight', pad_inches=.02)


def hsj_failure():
    beta = "1"
    noise = "stochastic"
    n_samples = "5"
    attacks = ['hsj', 'psj']
    repeats = [1, 1]
    flips = ['0.00', '0.05', '0.10'][:1]
    datasets = ['mnist', 'cifar10'][:1]
    for dataset in datasets:
        print(f'=========={dataset}=========')
        print('****\t' + '\t'.join(attacks))
        for flip in flips:
            print(flip, end='\t')
            for repeat, attack in zip(repeats, attacks):
                if dataset == 'mnist':
                    exp_name = f"{attack}_r_{repeat}_b_{beta}_{noise}_fp_{flip}_ns_{n_samples}"
                else:
                    exp_name = f"{dataset}_{attack}_r_{repeat}_b_{beta}_{noise}_fp_{flip}_ns_{n_samples}"
                raw = read_dump(exp_name)
                calls = np.median(raw['model_calls'][-1])
                dist = np.median(raw['border_distance'][-1])
                dist_40 = np.percentile(raw['border_distance'][-1], 40)
                dist_60 = np.percentile(raw['border_distance'][-1], 60)

                # print(f'({flip},{attack})\t', end='')
                print(f'{calls} {dist} {dist_40} {dist_60}', end='\t')
            print('')


def plot_distance_vs_rounds():
    plt.figure(figsize=(10, 7))
    dataset = 'mnist'
    attacks = ['hsj', 'psj', 'psj', 'psj', 'psj']
    noises = ['deterministic', 'deterministic', 'smoothing', 'cropping', 'dropout']
    labels = ['HSJ', 'PSJ', 'PSJ-Smoothing', 'PSJ-cropping', 'PSJ-dropout']
    noise_values = [None, None, 0.01, 27, 0.2]
    for i in range(len(attacks)):
        attack, noise, noise_value = attacks[i], noises[i], noise_values[i]
        sn = noise_value if noise == 'smoothing' else 0.01
        cs = noise_value if noise == 'cropping' else 26
        dr = noise_value if noise == 'dropout' else 0.5
        exp_name = f'{dataset}_{attack}_r_1_sn_{sn}_cs_{cs}_dr_{dr}_b_1_{noise}_fp_0.00_ns_100'
        raw = read_dump(exp_name)
        calls = np.median(raw['model_calls'].cpu().numpy(), axis=1)
        metric = 'border_distance'
        # metric = 'border_distance_smooth'
        dist = np.sqrt(np.median(raw[metric].cpu().numpy(), axis=1))
        dist_40 = np.sqrt(np.percentile(raw[metric].cpu().numpy(), 40, axis=1))
        dist_60 = np.sqrt(np.percentile(raw[metric].cpu().numpy(), 60, axis=1))
        print (dist_60)
        plt.plot(dist, label=f'{labels[i]}')
        plt.fill_between(range(0, len(dist_40)), dist_40, dist_60, alpha=0.2)
        # plt.plot(np.sqrt(dist_smooth), label=f'{labels[i]}: average of outputs')

    # plt.xscale('log')
    # plt.xlabel('smoothing noise ($\delta$)')
    plt.ylabel('border distance')
    plt.legend()
    plt.grid()
    plt.savefig(f'aistats/plots_aistats_rebuttal/border_distance_vs_rounds.pdf')


def plot_distance_vs_noise():
    plt.figure(figsize=(10, 7))
    dataset = 'mnist'
    noises = ['dropout', 'smoothing', 'cropping']
    dim = {'mnist': 784}
    attacks = ['hsj', 'psj']
    line_style = ['--', '-']
    labels = ['HSJ', 'PSJ']
    noise_rates = {
        'smoothing': ([0.01, 0.005, 0.001, 0.0005, 0.0001], 1),
        'dropout': ([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], 1),
        'cropping': ([25, 26, 27, 28], -1),
    }
    for noise in noises:
        epsilon = noise_rates[noise][0]
        for i, attack in enumerate(attacks):
            border_distance = []
            for eps in epsilon:
                sn = eps if noise == 'smoothing' else 0.01
                cs = eps if noise == 'cropping' else 26
                dr = eps if noise == 'dropout' else 0.5
                exp_name = f'{dataset}_{attack}_r_1_sn_{sn}_cs_{cs}_dr_{dr}_b_1_{noise}_fp_0.00_ns_100'
                raw = read_dump(exp_name)
                calls = np.median(raw['model_calls'].cpu().numpy(), axis=1)
                dist = np.median(raw['border_distance'].cpu().numpy(), axis=1)
                dist_smooth = np.median(raw['border_distance_smooth'].cpu().numpy(), axis=1)
                border_distance.append(dist[-1])

            ticks = np.array(epsilon)[::noise_rates[noise][1]]
            ticks = (ticks - np.min(ticks)) / (np.max(ticks) - np.min(ticks))
            plt.plot(ticks, np.sqrt(border_distance), line_style[i], label=f'{labels[i]}, {noise}')
        # plt.plot(np.sqrt(dist), label=f'{labels[i]}: without perturbation')
        # plt.plot(np.sqrt(dist_smooth), label=f'{labels[i]}: average of outputs')
        # plt.plot((np.sqrt(dist_smooth) - np.sqrt(dist)))
    # plt.xscale('log')
    plt.xlabel('noise ($\delta$)')
    plt.ylabel('median border distance')
    plt.legend()
    plt.grid()
    plt.savefig(f'aistats/plots_aistats_rebuttal/border_distance.pdf')

def get_smoothed_prob(x, noise, true_label, dataset, noise_rate):
    samples = 1000
    if dataset == 'cifar10':
        model = get_model(key='cifar10', dataset=dataset, noise='smoothing', smoothing_noise=noise)
    else:
        if noise == 'smoothing':
            model = get_model(key='mnist_noman', dataset=dataset, noise=noise, smoothing_noise=noise_rate)
        elif noise == 'dropout':
            model = get_model(key='mnist_noman', dataset=dataset, noise=noise, drop_rate=noise_rate)
        else:
            raise RuntimeError
    model.model = model.model.to(device)
    dim = [samples] + [1] * x.dim()
    x = x.unsqueeze(dim=0).repeat(*(dim))
    pred = model.ask_model(x)
    correct_pred = torch.sum(pred == true_label).float()
    return correct_pred / samples


def plot_adverarial_accuracy():
    use_memory = False
    plt.figure(figsize=(10, 7))
    dataset = 'mnist'
    noise = 'dropout'
    dim = {'mnist': 784}
    attacks = ['hsj', 'psj']
    line_style = ['--', '-']
    labels = ['HSJ', 'PSJ']
    noise_rates = {
        'smoothing': [0.01, 0.005, 0.001, 0.0005, 0.0001],
        'dropout': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    }
    epsilons = np.logspace(np.log10(0.5), np.log10(10), 501)
    if use_memory:
        memory = torch.load(open('aistats/plots_aistats_rebuttal/adversarial_acc.pkl', 'rb'), map_location=device)
    else:
        memory = {'epsilons': epsilons}
    for i, attack in enumerate(attacks):
        if not use_memory:
            memory[attack] = {}
        for n in noise_rates[noise]:
            sn = n if noise == 'smoothing' else 0.01
            cs = n if noise == 'cropping' else 26
            dr = n if noise == 'dropout' else 0.5
            print(attack, n)
            if not use_memory:
                exp_name = f'{dataset}_{attack}_r_1_sn_{sn}_cs_{cs}_dr_{dr}_b_1_{noise}_fp_0.00_ns_1'
                raw = read_dump(exp_name, raw=True)
                n_images = len(raw)
                AA = torch.zeros(size=(len(epsilons), n_images))
                for image in tqdm(range(n_images)):
                    diary = raw[image]
                    x_star = diary.original
                    x_t = diary.iterations[-1].bin_search
                    for e, eps in enumerate(epsilons):
                        x_adv = x_star + eps * (x_t - x_star) / torch.norm(x_t - x_star)
                        p_adv = get_smoothed_prob(x_adv, noise, diary.true_label, 'mnist', n)
                        AA[e, image] = p_adv
                memory[attack][n] = torch.mean(AA, dim=1)
            plt.plot(memory['epsilons'] / np.sqrt(dim[dataset]), memory[attack][n], line_style[i],
                     label=f'{labels[i]}($\delta =$ {n})')
    if not use_memory:
        torch.save(memory, open('aistats/plots_aistats_rebuttal/adversarial_acc.pkl', 'wb'))
    # plt.xscale('log')
    plt.ylabel('adversarial accuracy')
    plt.xlabel('epsilon')
    plt.legend()
    plt.grid()
    plt.savefig(f'aistats/plots_aistats_rebuttal/adversarial_acc_{noise}.pdf')


# estimate_repeat_in_hsj(beta=10)

# rc('text', usetex=True)
# rc('text.latex', preamble=[r'\usepackage{amsfonts}'])
# rc('font', size=14)  # default: 10 -> choose size depending on figsize
# rc('font', family='STIXGeneral')
# rc('legend', fontsize=12)
# plt.tight_layout(h_pad=0, w_pad=.5)

# attack='hsj_rep_psj_delta'
# dataset='mnist'
# estimate_repeat_in_hsj(beta_vs_repeats, dataset)
# hsj_failure()
# psj_vs_hsjr(best_repeat)
# conv_to_hsja()
plot_distance_vs_rounds()
plot_distance_vs_noise()
# plot_adverarial_accuracy()
