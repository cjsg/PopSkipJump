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
dataset = 'cifar10'
eps = list(range(1, 6))

def read_dump(path, raw=False):
    if raw:
        filepath = f'{OUT_DIR}/{path}/raw_data.pkl'
    else:
        filepath = f'{OUT_DIR}/{path}/crunched_new.pkl'
    raw = torch.load(open(filepath, 'rb'), map_location=device)
    return raw


def plot_distance_vs_rounds():
    plt.figure(figsize=(10, 7))
    n_samples = 100 if dataset == 'mnist' else 50
    attacks = ['hsj', 'psj', 'psj', 'psj', 'psj']
    noises = ['deterministic', 'deterministic', 'smoothing', 'cropping', 'dropout']
    labels = ['HSJ-deterministic', 'PSJ-deterministic', 'PSJ-smoothing', 'PSJ-cropping', 'PSJ-dropout']
    noise_values = [None, None, 0.001, 26, 0.5] if dataset == 'mnist' else [None, None, 0.005, 30, 0.02]
    for i in range(len(attacks)):
        attack, noise, noise_value = attacks[i], noises[i], noise_values[i]
        sn = noise_value if noise == 'smoothing' else 0.01
        cs = noise_value if noise == 'cropping' else 26
        dr = noise_value if noise == 'dropout' else 0.5
        if noise in ['cropping'] and dataset == 'cifar10':
            exp_name = f'{dataset}_{attack}_r_1_sn_{sn}_cs_{cs}_dr_{dr}_b_1_{noise}_fp_0.00_ns_{20}'
        else:
            exp_name = f'{dataset}_{attack}_r_1_sn_{sn}_cs_{cs}_dr_{dr}_b_1_{noise}_fp_0.00_ns_{n_samples}'
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
    ref = 'a' if dataset == 'mnist' else 'c'
    plt.xlabel(f'({ref}) gradient steps')
    plt.ylabel('median border distance')
    title_height = 0.203 if dataset == 'mnist' else 0.16
    plt.text(12, title_height, dataset.upper())
    plt.legend()
    plt.grid()
    plt.savefig(f'aistats/plots_aistats_rebuttal/{dataset}_border_distance_vs_rounds.pdf', bbox_inches='tight')


def plot_distance_vs_noise():
    plt.figure(figsize=(10, 7))
    n_samples = 100 if dataset == 'mnist' else 50
    noises = ['smoothing', 'cropping', 'dropout']
    attacks = ['hsj', 'psj']
    line_style = [(0,(5,8)), (0, ())]
    labels = ['HSJ', 'PSJ']
    noise_rates_mnist = {
        'smoothing': ([0.01, 0.005, 0.001, 0.0005, 0.0001], 1),
        'dropout': ([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], 1),
        'cropping': ([25, 26, 27, 28], -1),
    }
    noise_rates_cifar = {
        'smoothing': ([0.005, 0.001, 0.0005, 0.0001, 0.00], 1),
        'dropout': ([0.0, 0.01, 0.02], 1),
        'cropping': ([30, 31, 32], -1),
    }
    noise_rates = noise_rates_mnist if dataset == 'mnist' else noise_rates_cifar
    for kk, noise in enumerate(noises):
        epsilon = noise_rates[noise][0]
        for i, attack in enumerate(attacks):
            border_distance = []
            border_distance_40 = []
            border_distance_60 = []
            for eps in epsilon:
                sn = eps if noise == 'smoothing' else 0.01
                cs = eps if noise == 'cropping' else 26
                dr = eps if noise == 'dropout' else 0.5
                if noise in ['cropping', 'dropout'] and dataset == 'cifar10':
                    exp_name = f'{dataset}_{attack}_r_1_sn_{sn}_cs_{cs}_dr_{dr}_b_1_{noise}_fp_0.00_ns_{20}'
                else:
                    exp_name = f'{dataset}_{attack}_r_1_sn_{sn}_cs_{cs}_dr_{dr}_b_1_{noise}_fp_0.00_ns_{n_samples}'
                raw = read_dump(exp_name)
                metric = 'border_distance'
                # metric = 'border_distance_smooth'
                dist = np.sqrt(np.median(raw[metric].cpu().numpy(), axis=1))
                dist_40 = np.sqrt(np.percentile(raw[metric].cpu().numpy(), 40, axis=1))
                dist_60 = np.sqrt(np.percentile(raw[metric].cpu().numpy(), 60, axis=1))
                border_distance.append(dist[-1])
                border_distance_40.append(dist_40[-1])
                border_distance_60.append(dist_60[-1])

            ticks = np.array(epsilon)[::noise_rates[noise][1]]
            ticks = (ticks - np.min(ticks)) / (np.max(ticks) - np.min(ticks))
            plt.plot(ticks, border_distance, linestyle=line_style[i], color=f'C{kk+2}', label=f'{labels[i]}, {noise}')
            plt.fill_between(ticks, border_distance_40, border_distance_60, color=f'C{kk+2}', alpha=0.2)

        # plt.plot(np.sqrt(dist), label=f'{labels[i]}: without perturbation')
        # plt.plot(np.sqrt(dist_smooth), label=f'{labels[i]}: average of outputs')
        # plt.plot((np.sqrt(dist_smooth) - np.sqrt(dist)))
    # plt.xscale('log')
    height = 0.034 if dataset == 'mnist' else -0.011
    title_height = 0.265 if dataset == 'mnist' else 0.081
    # plt.text(-0.05, height, 'no noise')
    # plt.text(0.9, height, 'high noise')
    ref = 'b' if dataset == 'mnist' else 'd'
    # plt.text(0.48, height, f'({ref})')
    plt.text(0.40, title_height, dataset.upper())

    plt.xticks([0, 1], ['none', 'high'])
    # plt.tick_params(
    #     axis='x',  # changes apply to the x-axis
    #     which='both',  # both major and minor ticks are affected
    #     bottom=False,  # ticks along the bottom edge are off
    #     top=False,  # ticks along the top edge are off
    #     labelbottom=False)  # labels along the bottom edge are off
    plt.xlabel(f'({ref}) noise level')
    plt.ylabel('median border distance')
    plt.grid()
    plt.legend()
    plt.savefig(f'aistats/plots_aistats_rebuttal/{dataset}_border_distance_vs_noise.pdf', bbox_inches='tight')

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

rc('text', usetex=True)
rc('text.latex', preamble=[r'\usepackage{amsfonts}'])
rc('font', size=26)  # default: 10 -> choose size depending on figsize
rc('font', family='STIXGeneral')
rc('legend', fontsize=20)
plt.tight_layout(h_pad=0, w_pad=0)

# attack='hsj_rep_psj_delta'
# dataset='mnist'
# estimate_repeat_in_hsj(beta_vs_repeats, dataset)
# hsj_failure()
# psj_vs_hsjr(best_repeat)
# conv_to_hsja()
plot_distance_vs_rounds()
plot_distance_vs_noise()
# plot_adverarial_accuracy()
