import matplotlib.pylab as plt
import numpy as np
import torch
from matplotlib import rc

from img_utils import get_device

OUT_DIR = 'thesis'
PLOTS_DIR = f'{OUT_DIR}/plots_experiments/'
device = get_device()


def read_dump(path, raw=False, aa=False):
    if raw:
        filepath = f'{OUT_DIR}/{path}/raw_data.pkl'
    elif aa:
        filepath = f'{OUT_DIR}/{path}/crunched_aa.pkl'
    else:
        filepath = f'{OUT_DIR}/{path}/crunched.pkl'
    raw = torch.load(open(filepath, 'rb'), map_location=device)
    return raw


noise_level = {'mnist': {
    'bayesian': (['5', '2', '1.50', '1'], [0, 0.2, 0.5, 0.67, 1]),
    'smoothing': (['0.0001', '0.001', '0.01'], [0, 0.1, 0.45, 1]),
    'cropping': (['27', '26', '25'], [0, 0.33, 0.67, 1]),
    'dropout': (['0.1', '0.3', '0.5'], [0, 0.1, 0.3, 0.5]),
}, 'cifar10': {
    'bayesian': (['5', '1.50', '1'], [0, 0.2, 0.67, 1]),
    'smoothing': (['0.0001', '0.0005', '0.001', '0.005'], [0, 0.1, 0.2, 0.45, 0.55]),
    'cropping': (['31', '30'], [0, 0.5, 1]),
    'dropout': (['0.01', '0.02', '0.03'], [0, 0.01, 0.02, 0.03])
}}

# noise_rates_mnist = {
#     'bayesian': (['1', '1.50', '2', '5'], -1),
#     'smoothing': ([0.01, 0.005, 0.001, 0.0005, 0.0001], 1),
#     'dropout': ([0.1, 0.2, 0.3, 0.4, 0.5], 1),
#     'cropping': ([25, 26, 27, 28], -1),
# }
# noise_rates_cifar = {
#     'bayesian': (['1', '1.50', '5'], -1),
#     'smoothing': ([0.005, 0.001, 0.0005, 0.0001], 1),
#     'dropout': ([0.01, 0.02, 0.03], 1),
#     'cropping': ([30, 31, 32], -1),
# }
best_noises = {'mnist': {
    'bayesian': '1.50',
    'smoothing': 0.01,
    'dropout': 0.5,
    'cropping': 25
}, 'cifar10': {
    'bayesian': '1.50',
    'smoothing': 0.005,
    'dropout': 0.03,
    'cropping': 30
}}
props = {'ha': 'center', 'va': 'center'}


def get_label(noise, level, attack=None, dataset=None):
    if noise == 'bayesian':
        if level == 'det':
            return attack.upper() + ': T=0 (det)'
        t = 1.0 / float(level)
        return 'PSJ: T=%.2f' % t
    if noise == 'smoothing':
        if level == 'det':
            return attack.upper() + ': $\\sigma$=0 (det)'
        return 'PSJ: $\\sigma$={:.0e}'.format(float(level))
    if noise == 'dropout':
        if level == 'det':
            return attack.upper() + ': $\\alpha$=0 (det)'
        return 'PSJ: $\\alpha$=%.2f' % float(level)
    if noise == 'cropping':
        if level == 'det':
            if dataset == 'mnist':
                return attack.upper() + ': s=28 (det)'
            elif dataset == 'cifar10':
                return attack.upper() + ': s=32 (det)'
        return f'PSJ: s={int(level)}'


def grid():
    rc('text', usetex=True)
    rc('text.latex', preamble=[r'\usepackage{amsfonts}'])
    rc('font', size=12)  # default: 10 -> choose size depending on figsize
    rc('font', family='STIXGeneral')
    rc('legend', fontsize=12)
    plt.tight_layout(h_pad=0, w_pad=0)
    n_images = 100
    image_path = f'thesis/plots_paper/grid_{n_images}.pdf'
    datasets = ['mnist', 'cifar10']
    noises = ['bayesian', 'dropout', 'smoothing', 'cropping']
    noise_names = ['logit sampling', 'dropout', 'adversarial smoothing', 'random cropping']
    fig3 = plt.figure(figsize=(20, 14))
    gs = fig3.add_gridspec(160, 260)
    count = 1
    for nn, noise in enumerate(noises):
        for dd, dataset in enumerate(datasets):
            ax0 = fig3.add_subplot(gs[nn * 37:nn * 37 + 30, 131 * dd:131 * dd + 59])
            ax1 = fig3.add_subplot(gs[nn * 37:nn * 37 + 30, 131 * dd + 60:131 * dd + 119], sharey=ax0)
            # ax0 = plt.subplot(len(noises), 2 * len(datasets), count)
            props = {'ha': 'center', 'va': 'center'}
            if dd == 0:
                props = {'ha': 'center', 'va': 'center'}
                plt.text(-0.25, 0.5, noise_names[nn], props, rotation=90, transform=ax0.transAxes, fontsize=14)
            if nn == 0:
                ax0.text(0.5, 1.05, dataset.upper(), props, transform=ax0.transAxes, fontsize=16)
                ax1.text(0.5, 1.05, dataset.upper(), props, transform=ax1.transAxes, fontsize=16)

            for attack in ['psj', 'hsj']:
                exp_name = f'{dataset}_{attack}_r_1_sn_0.01_cs_26_dr_0.5_dm_l2_b_1_deterministic_fp_0.00_ns_{n_images}'
                raw = read_dump(exp_name)
                metric = 'border_distance'
                D = raw[metric]
                C = raw['model_calls']
                if dataset == 'cifar10':
                    D = torch.cat([D[:, :26], D[:, 27:57], D[:, 59:61], D[:, 62:]], dim=1)
                    C = torch.cat([C[:, :26], C[:, 27:57], C[:, 59:61], C[:, 62:]], dim=1)
                dist = np.median(D, axis=1)
                calls = np.median(C, axis=1)
                perc_40 = np.percentile(D, 40, axis=1)
                perc_60 = np.percentile(D, 60, axis=1)
                ax0.plot(dist, label=get_label(noise, 'det', attack=attack, dataset=dataset))
                ax0.fill_between(range(len(perc_40)), perc_40, perc_60, alpha=0.1)
                ax1.plot(calls, dist, label=get_label(noise, 'det', attack=attack, dataset=dataset))
            for level in noise_level[dataset][noise][0]:
                b, sn, cs, dr = '1', '0.01', '26', '0.5'
                if noise == 'bayesian':
                    b = level
                elif noise == 'smoothing':
                    sn = level
                elif noise == 'cropping':
                    cs = level
                elif noise == 'dropout':
                    dr = level
                else:
                    raise RuntimeError
                exp_name = f'{dataset}_psj_r_1_sn_{sn}_cs_{cs}_dr_{dr}_dm_l2_b_{b}_{noise}_fp_0.00_ns_{n_images}'
                raw = read_dump(exp_name)
                metric = 'border_distance'
                D = raw[metric]
                C = raw['model_calls']
                if dataset == 'cifar10':
                    D = torch.cat([D[:, :26], D[:, 27:57], D[:, 59:61], D[:, 62:]], dim=1)
                    C = torch.cat([C[:, :26], C[:, 27:57], C[:, 59:61], C[:, 62:]], dim=1)
                dist = np.median(D, axis=1)
                calls = np.median(C, axis=1)
                perc_40 = np.percentile(D, 40, axis=1)
                perc_60 = np.percentile(D, 60, axis=1)
                ax0.plot(dist, label=get_label(noise, level))
                ax0.fill_between(range(len(perc_40)), perc_40, perc_60, alpha=0.1)
                ax1.plot(calls, dist, label=get_label(noise, level))
            ax0.legend()
            ax0.grid(axis='y')
            ax0.set_ylabel('border distance')
            ax0.set_xlabel('rounds')
            ax1.grid(axis='y')
            ax1.set_xlabel('median model calls')
            ax1.set_xscale('log')

            plt.setp(ax1.get_yticklabels(), visible=False)
            count += 2

    plt.savefig(image_path, bbox_inches='tight')


def noise():
    rc('text', usetex=True)
    rc('text.latex', preamble=[r'\usepackage{amsfonts}'])
    rc('font', size=10)  # default: 10 -> choose size depending on figsize
    rc('font', family='STIXGeneral')
    rc('legend', fontsize=10)
    plt.tight_layout(h_pad=0, w_pad=0)

    n_samples = 100
    noises = ['smoothing', 'smoothing', 'cropping', 'dropout']
    attacks = ['hsj', 'psj']
    line_style = [(0, (5, 8)), (0, ())]
    labels = ['HSJ', 'PSJ']
    datasets = ['mnist', 'cifar10']
    plt.figure(figsize=(20, 4))
    for dd, dataset in enumerate(datasets):
        noise_rates = noise_rates_mnist if dataset == 'mnist' else noise_rates_cifar
        ax0 = plt.subplot(1, 4, 2 * dd + 1)
        ax1 = plt.subplot(1, 4, 2 * dd + 2)
        for kk, noise in enumerate(noises):
            epsilon = noise_rates[noise][0]
            for i, attack in enumerate(attacks):
                metric = 'border_distance'
                # metric = 'border_distance_smooth'
                border_distance = []
                border_distance_40 = []
                border_distance_60 = []
                for eps in epsilon:
                    sn = eps if noise == 'smoothing' else 0.01
                    cs = eps if noise == 'cropping' else 26
                    dr = eps if noise == 'dropout' else 0.5
                    b = eps if noise == 'bayesian' else '1'
                    exp_name = f'{dataset}_{attack}_r_1_sn_{sn}_cs_{cs}_dr_{dr}_dm_l2_b_{b}_{noise}_fp_0.00_ns_{n_samples}'
                    raw = read_dump(exp_name)

                    D = raw[metric]
                    if dataset == 'cifar10':
                        D = torch.cat([D[:, :26], D[:, 27:57], D[:, 59:61], D[:, 62:]], dim=1)
                    dist = np.median(D, axis=1)
                    dist_40 = np.percentile(D, 40, axis=1)
                    dist_60 = np.percentile(D, 60, axis=1)
                    border_distance.append(dist[-1])
                    border_distance_40.append(dist_40[-1])
                    border_distance_60.append(dist_60[-1])
                    if eps == best_noises[dataset][noise] and attack == 'psj':
                        ax0.plot(dist, label=f'PSJ ({noise})')
                        ax0.fill_between(range(len(dist)), dist_40, dist_60, alpha=0.2)
                ticks = np.array([float(e) for e in epsilon])[::noise_rates[noise][1]]
                ticks = (ticks - np.min(ticks)) / (np.max(ticks) - np.min(ticks))
                ax1.plot(ticks, border_distance, linestyle=line_style[i], color=f'C{kk + 2}',
                         label=f'{labels[i]}, {noise}')
                ax1.fill_between(ticks, border_distance_40, border_distance_60, color=f'C{kk + 2}', alpha=0.2)
        ref = ['a', 'b', 'c', 'd']
        # plt.subplot(1, 4, 2*dd + 2)
        ax0.set_xlabel(f'({ref[2 * dd]}) gradient steps')
        ax0.set_ylabel(f'median border dist ({dataset})')
        ax0.grid()
        ax0.legend()
        ax1.set_xticks([0, 1])  # values
        ax1.set_xticklabels(['none', 'high'])  # labels
        ax1.set_xlabel(f'({ref[2 * dd + 1]}) noise level')
        ax1.set_ylabel(f'median border dist ({dataset})')
        ax1.grid()
        ax1.legend()
    image_path = f'thesis/plots_paper/noise_{n_samples}.pdf'
    plt.savefig(image_path, bbox_inches='tight')


def fig3_2lines(line=None):
    def extract_dist(exp_name, metric):
        raw = read_dump(exp_name)
        D = raw[metric]
        if dataset == 'cifar10':
            D = torch.cat([D[:, :26], D[:, 27:57], D[:, 59:61], D[:, 62:]], dim=1)
        dist = np.median(D, axis=1)
        dist_40 = np.percentile(D, 40, axis=1)
        dist_60 = np.percentile(D, 60, axis=1)
        return dist, dist_40, dist_60

    rc('text', usetex=True)
    rc('text.latex', preamble=[r'\usepackage{amsfonts}'])
    rc('font', size=15)  # default: 10 -> choose size depending on figsize
    rc('font', family='STIXGeneral')
    rc('legend', fontsize=12)
    plt.tight_layout(h_pad=0, w_pad=0)
    n_images = 100
    ref = ['a', 'b', 'c', 'd']
    if line is None:
        image_path = f'thesis/plots_paper/fig3_bothlines_{n_images}.pdf'
    else:
        image_path = f'thesis/plots_paper/fig3_line{line}_{n_images}.pdf'
    datasets = ['mnist', 'cifar10']
    noises = ['bayesian', 'dropout', 'smoothing', 'cropping']
    noise_names_2 = ['logit sampling', 'dropout', 'adversarial smoothing', 'random cropping']
    noise_names_1 = ['logit', 'dropout', 'smoothing', 'cropping']
    fig3 = plt.figure(figsize=(20, 7))
    gs = fig3.add_gridspec(80, 300)
    attacks = ['hsj', 'psj']
    # line_style = [(0, (5, 8)), (0, ())]
    line_style = ['--', (0, ())]
    labels = ['HSJ', 'PSJ']
    # plt.figure(figsize=(20, 4))
    if line == 1 or line is None:
        rc('font', size=18)  # default: 10 -> choose size depending on figsize
        label_fontsize = 22
        nn = 0
        for dd, dataset in enumerate(datasets):
            # noise_rates = noise_rates_mnist if dataset == 'mnist' else noise_rates_cifar
            # ax0 = plt.subplot(1, 4, 2*dd + 1)
            # ax1 = plt.subplot(1, 4, 2*dd + 2)
            ax0 = fig3.add_subplot(gs[nn * 37:nn * 37 + 30, 145 * dd:145 * dd + 60])
            ax1 = fig3.add_subplot(gs[nn * 37:nn * 37 + 30, 145 * dd + 71:145 * dd + 131])
            if nn == 0:
                height = 1.10
                ax0.text(0.5, height, dataset.upper(), props, transform=ax0.transAxes, fontsize=22)
                ax1.text(0.5, height, dataset.upper(), props, transform=ax1.transAxes, fontsize=22)
            for kk, noise in enumerate(noises):
                epsilon = noise_level[dataset][noise][0]
                for i, attack in enumerate(attacks):
                    metric = 'border_distance'
                    # metric = 'border_distance_smooth'
                    border_distance = []
                    border_distance_40 = []
                    border_distance_60 = []
                    exp_name = f'{dataset}_{attack}_r_1_sn_0.01_cs_26_dr_0.5_dm_l2_b_1_deterministic_fp_0.00_ns_{n_images}'
                    dist, dist_40, dist_60 = extract_dist(exp_name, metric)
                    border_distance.append(dist[-1])
                    border_distance_40.append(dist_40[-1])
                    border_distance_60.append(dist_60[-1])
                    for eps in epsilon:
                        sn = eps if noise == 'smoothing' else 0.01
                        cs = eps if noise == 'cropping' else 26
                        dr = eps if noise == 'dropout' else 0.5
                        b = eps if noise == 'bayesian' else '1'
                        exp_name = f'{dataset}_{attack}_r_1_sn_{sn}_cs_{cs}_dr_{dr}_dm_l2_b_{b}_{noise}_fp_0.00_ns_{n_images}'
                        dist, dist_40, dist_60 = extract_dist(exp_name, metric)
                        border_distance.append(dist[-1])
                        border_distance_40.append(dist_40[-1])
                        border_distance_60.append(dist_60[-1])
                        if eps == str(best_noises[dataset][noise]) and attack == 'psj':
                            ax0.plot(dist, color=f'C{kk + 2}', label=f'PSJ ({noise_names_1[kk]})')
                            ax0.fill_between(range(len(dist)), dist_40, dist_60, color=f'C{kk + 2}', alpha=0.2)
                    ticks = noise_level[dataset][noise][1]
                    ticks = (ticks - np.min(ticks)) / (np.max(ticks) - np.min(ticks))
                    if attack == 'hsj':
                        label = f'{labels[i]}'
                    else:
                        label = f'{labels[i]}\t({noise_names_1[kk]})'
                    ax1.plot(ticks, border_distance, linestyle=line_style[i], color=f'C{kk + 2}',
                             label=label)
                    ax1.fill_between(ticks, border_distance_40, border_distance_60, color=f'C{kk + 2}', alpha=0.2)
            # plt.subplot(1, 4, 2*dd + 2)
            ax0.set_xlabel(f'({ref[2 * dd]}) iterations', fontsize=label_fontsize)
            if dd==0:
                ax0.set_ylabel(f'median border dist', fontsize=label_fontsize)
            ax0.grid()
            # ax0.legend()
            from matplotlib.ticker import FormatStrFormatter
            ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax1.set_xticks([0, 1])  # values
            ax1.set_xticklabels(['none', 'high'])  # labels
            ax1.set_xlabel(f'({ref[2 * dd + 1]}) noise level', fontsize=label_fontsize)
            # ax1.set_ylabel(f'median border dist', fontsize=label_fontsize)
            ax1.grid()
            if dd == 1:
                handles, labels = ax1.get_legend_handles_labels()
                new_handles, new_labels = [], []
                for q in [0, 4, 2, 6, 1, 5, 3, 7]:
                    new_handles.append(handles[q])
                    new_labels.append(labels[q])

                # ax1.legend(loc='upper center', bbox_to_anchor=(1.11, 1.2), ncol=8, fancybox=True, handletextpad=1)
                ax1.legend(new_handles, new_labels, loc='upper center', bbox_to_anchor=(-0.63, 1),
                           ncol=2, fancybox=True, columnspacing=0.4, handleheight=1.2)
                # ax1.legend(loc='upper center', bbox_to_anchor=(1.1, 1.2), ncol=8, fancybox=True)
    if line == 2 or line is None:
        rc('font', size=16)  # default: 10 -> choose size depending on figsize
        label_fontsize = 20
        if line == 2:
            nn = 0
        else:
            nn = 1
        noise = 'dropout'
        for dd, dataset in enumerate(datasets):
            if line is None:
                ax0 = fig3.add_subplot(gs[nn * 37:nn * 37 + 30, 145 * dd:145 * dd + 60])
                ax1 = fig3.add_subplot(gs[nn * 37:nn * 37 + 30, 145 * dd + 72:145 * dd + 132])
            else:
                ax0 = fig3.add_subplot(gs[nn * 37:nn * 37 + 30, 138 * dd:138 * dd + 60])
                ax1 = fig3.add_subplot(gs[nn * 37:nn * 37 + 30, 138 * dd + 62:138 * dd + 122], sharey=ax0)

            if nn == 0:
                ax0.text(0.5, 1.09, dataset.upper(), props, transform=ax0.transAxes, fontsize=21)
                ax1.text(0.5, 1.09, dataset.upper(), props, transform=ax1.transAxes, fontsize=21)
            for attack in ['hsj', 'psj']:
                exp_name = f'{dataset}_{attack}_r_1_sn_0.01_cs_26_dr_0.5_dm_l2_b_1_deterministic_fp_0.00_ns_{n_images}'
                raw = read_dump(exp_name)
                metric = 'border_distance'
                D = raw[metric]
                C = raw['model_calls']
                if dataset == 'cifar10':
                    D = torch.cat([D[:, :26], D[:, 27:57], D[:, 59:61], D[:, 62:]], dim=1)
                    C = torch.cat([C[:, :26], C[:, 27:57], C[:, 59:61], C[:, 62:]], dim=1)
                dist = np.median(D, axis=1)
                calls = np.median(C, axis=1)
                perc_40 = np.percentile(D, 40, axis=1)
                perc_60 = np.percentile(D, 60, axis=1)
                if attack == 'hsj':
                    ax0.plot(dist, label=get_label(noise, 'det', attack=attack, dataset=dataset), linestyle='--')
                    ax0.fill_between(range(len(perc_40)), perc_40, perc_60, alpha=0.1)
                    ax1.plot(calls, dist, label=get_label(noise, 'det', attack=attack, dataset=dataset), linestyle='--')
                else:
                    ax0.plot(dist, label=get_label(noise, 'det', attack=attack, dataset=dataset))
                    ax0.fill_between(range(len(perc_40)), perc_40, perc_60, alpha=0.1)
                    ax1.plot(calls, dist, label=get_label(noise, 'det', attack=attack, dataset=dataset))

            for level in noise_level[dataset][noise][0]:
                b, sn, cs, dr = '1', '0.01', '26', '0.5'
                if noise == 'bayesian':
                    b = level
                elif noise == 'dropout':
                    dr = level
                else:
                    raise RuntimeError
                exp_name = f'{dataset}_psj_r_1_sn_{sn}_cs_{cs}_dr_{dr}_dm_l2_b_{b}_{noise}_fp_0.00_ns_{n_images}'
                raw = read_dump(exp_name)
                metric = 'border_distance'
                D = raw[metric]
                C = raw['model_calls']
                if dataset == 'cifar10':
                    D = torch.cat([D[:, :26], D[:, 27:57], D[:, 59:61], D[:, 62:]], dim=1)
                    C = torch.cat([C[:, :26], C[:, 27:57], C[:, 59:61], C[:, 62:]], dim=1)
                dist = np.median(D, axis=1)
                calls = np.median(C, axis=1)
                perc_40 = np.percentile(D, 40, axis=1)
                perc_60 = np.percentile(D, 60, axis=1)
                ax0.plot(dist, label=get_label(noise, level))
                ax0.fill_between(range(len(perc_40)), perc_40, perc_60, alpha=0.1)
                ax1.plot(calls, dist, label=get_label(noise, level))
            ax0.legend(fontsize=13)
            ax0.grid(axis='y')
            ax0.set_ylabel('median border dist', fontsize=label_fontsize)
            ax1.grid(axis='y')
            if line is None:
                ax1.set_ylabel(f'median border dist', fontsize=label_fontsize)
                ax0.set_xlabel('iterations')
                ax1.set_xlabel('median model calls', fontsize=label_fontsize)
            else:
                ax0.set_xlabel(f'({ref[2 * dd]}) iterations', fontsize=label_fontsize)
                ax1.set_xlabel(f'({ref[2 * dd + 1]}) median model calls', fontsize=label_fontsize)
            ax1.set_xscale('log')
            if line == 2:
                plt.setp(ax1.get_yticklabels(), visible=False)

    plt.savefig(image_path, bbox_inches='tight')


def delta(dataset='cifar10'):
    n_images = 100
    image_path = f'thesis/plots_paper/delta_{dataset}_{n_images}.pdf'
    tfs = {
        'mnist': [784, 500, 100, 50, 10, 1],
        'cifar10': [3072, 1000, 500, 100, 50, 10, 1],
    }
    rc('text', usetex=True)
    rc('text.latex', preamble=[r'\usepackage{amsfonts}'])
    rc('font', size=22)  # default: 10 -> choose size depending on figsize
    rc('font', family='STIXGeneral')
    rc('legend', fontsize=22)
    plt.tight_layout(h_pad=0, w_pad=0)
    fig = plt.figure(figsize=(7, 5))

    for tf in tfs[dataset]:
        exp_name = f'{dataset}_hsj_rep_psj_delta_r_1_tf_{tf}_sn_0.01_cs_26_dr_0.5_dm_l2_b_1_deterministic_fp_0.00_ns_{n_images}'
        raw = read_dump(exp_name)
        metric = 'border_distance'
        D = raw[metric]
        if dataset == 'cifar10':
            D = torch.cat([D[:, :57], D[:, 58:]], dim=1)
        dist = np.median(D, axis=1)
        # perc_40 = np.percentile(raw[metric], 40, axis=1)
        # perc_60 = np.percentile(raw[metric], 60, axis=1)
        beta = tf * 1.0 / tfs[dataset][0]
        if tf == 1:
            if dataset == 'mnist':
                label = '$\\beta$=$d^{-1}$=1.28e-3'
            elif dataset == 'cifar10':
                label = '$\\beta$=$d^{-1}$=3.25e-4'
        else:
            label = '$\\beta$=%.3f' % float(beta)
        plt.plot(dist, label=label)
    if dataset == 'mnist':
        # plt.legend()
        plt.legend(bbox_to_anchor=(1.02, 0.95), loc='upper right')
    else:
        plt.legend(bbox_to_anchor=(1, 0.95), loc='upper right')

    plt.text(0.45, 0.95, dataset.upper(), props, transform=fig.axes[0].transAxes, fontsize=28)
    plt.xlabel('iterations', fontsize=32)
    plt.ylabel('median border dist', fontsize=32)
    plt.savefig(image_path, bbox_inches='tight')


def hsj_vs_psj():
    params = {'mnist': {
        'bayesian': {
            # 1: [1, 10, 100, 1000, 10000, 100000],
            2: [1, 10, 100, 1000, 10000, 100000, 1000000, 5000000],
            5: [1, 10, 100, 1000, 10000, 100000, 1000000],
            10: [1, 10, 100, 1000, 10000, 100000],
            20: [1, 10, 100, 1000, 10000, 100000],
            50: [1, 10, 100, 1000, 10000]
        },
        # 'smoothing': {
        #     0.01: [1, 5, 10, 50, 100],
        #     0.005: [1, 5, 10, 50, 100],
        #     0.001: [1, 5, 10, 50, 100],
        #     0.0005: [1, 5, 10, 50, 100],
        #     0.0001: [1, 5, 10, 50, 100],
        # },
        # 'dropout': {
        #     0.1: [1, 5, 10, 50],
        #     0.2: [1, 5, 10, 50],
        #     0.3: [1, 5, 10, 100, 1000],
        #     0.4: [1, 5, 10, 100, 1000],
        #     0.5: [1, 5, 10, 100, 1000],
        # }
    }, 'cifar10': {
        'bayesian': {
            2: [1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 40000000],
            5: [1, 10, 100, 1000, 10000, 100000, 1000000, 4000000, 10000000],
            10: [1, 10, 100, 1000, 10000, 100000, 1000000],
            20: [1, 10, 100, 1000, 10000, 100000, 1000000],
            50: [1, 10, 100, 1000, 10000, 100000, 1000000],
            # 2: [100, 1000],
            # 5: [100, 1000],
            # 10: [100, 1000],
        },
    }}
    for dataset in params.keys():
        n_images = 100 if dataset == 'mnist' else 50
        for noise in params[dataset].keys():
            b, sn, cs, dr = 1, 0.01, 26, 0.5
            for noise_level in params[dataset][noise].keys():
                if noise == 'bayesian':
                    b = noise_level
                elif noise == 'smoothing':
                    sn = noise_level
                elif noise == 'cropping':
                    cs = noise_level
                elif noise == 'dropout':
                    dr = noise_level
                reps = params[dataset][noise][noise_level]
                exp_name = f'{dataset}_psj_r_1_sn_{sn}_cs_{cs}_dr_{dr}_dm_l2_b_{b}_{noise}_fp_0.00_ns_{n_images}'
                raw = read_dump(exp_name)
                data = raw['border_distance'][-1, :]
                dist = np.median(data)
                dist_40 = np.percentile(data, 40)
                dist_60 = np.percentile(data, 60)
                h, h_40, h_60 = [], [], []
                repsd, hd, hd_40, hd_60 = [], [], [], []
                for rep in reps:
                    exp_name = f'{dataset}_hsj_rep_r_{rep}_sn_{sn}_cs_{cs}_dr_{dr}_dm_l2_b_{b}_{noise}_fp_0.00_ns_{n_images}'
                    raw = read_dump(exp_name)
                    data = raw['border_distance'][-1, :]
                    h.append(np.median(data))
                    h_40.append(np.percentile(data, 40))
                    h_60.append(np.percentile(data, 60))

                    exp_name = f'{dataset}_hsj_rep_del_r_{rep}_sn_{sn}_cs_{cs}_dr_{dr}_dm_l2_b_{b}_{noise}_fp_0.00_ns_{n_images}'
                    try:
                        raw = read_dump(exp_name)
                        data = raw['border_distance'][-1, :]
                        repsd.append(rep)
                        hd.append(np.median(data))
                        hd_40.append(np.percentile(data, 40))
                        hd_60.append(np.percentile(data, 60))
                    except:
                        pass
                plt.figure()
                plt.plot(reps, h, label='hsj')
                plt.fill_between(reps, h_40, h_60, alpha=0.2)
                plt.plot(repsd, hd, label='hsj (psj delta)')
                plt.fill_between(repsd, hd_40, hd_60, alpha=0.2)
                plt.plot(reps, [dist] * len(reps), label='psj')
                plt.fill_between(reps, [dist_40] * len(reps), [dist_60] * len(reps), alpha=0.2)
                plt.legend()
                plt.xscale('log')
                plt.yscale('log')
                plt.savefig(f'thesis/plots_paper/zzz_{dataset}_{noise}_{noise_level}.pdf')


def fig5():
    rc('text', usetex=True)
    rc('text.latex', preamble=[r'\usepackage{amsfonts}'])
    rc('font', size=24)  # default: 10 -> choose size depending on figsize
    rc('font', family='STIXGeneral')
    rc('legend', fontsize=28)
    plt.tight_layout(h_pad=0, w_pad=0)
    data = {'mnist': {
        'bayesian': {2: (5000000, 10000), 5: (1000000, 1000), 10: (100000, 200), 20: (50000, 100), 50: (10000, 10)},
    }, 'cifar10': {
        'bayesian': {2: (40000000, 1000), 5: (4000000, 100), 10: (1000000, 50), 20: (500000, 10), 50: (100000, 5)}
    },
    }

    for dataset in data.keys():
        n_images = 100 if dataset == 'mnist' else 50
        fig = plt.figure(figsize=(7, 5))
        for noise in data[dataset].keys():
            z, y, y_40, y_60 = [], [], [], []
            y_delta, y_delta_40, y_delta_60 = [], [], []
            exp_name = f'{dataset}_psj_r_1_sn_0.01_cs_26_dr_0.5_dm_l2_b_1_deterministic_fp_0.00_ns_{n_images}'
            raw = read_dump(exp_name)
            x = np.median(raw['model_calls'][-1, :])
            x_40 = np.percentile(raw['model_calls'][-1, :], 40)
            x_60 = np.percentile(raw['model_calls'][-1, :], 60)
            exp_name = f'{dataset}_hsj_rep_r_1_sn_0.01_cs_26_dr_0.5_dm_l2_b_1_deterministic_fp_0.00_ns_{n_images}'
            raw = read_dump(exp_name)
            calls = raw['model_calls'][-1, :]
            exp_name = f'{dataset}_hsj_rep_del_r_1_sn_0.01_cs_26_dr_0.5_dm_l2_b_1_deterministic_fp_0.00_ns_{n_images}'
            raw = read_dump(exp_name)
            calls_delta = raw['model_calls'][-1, :]
            z.append(0)
            y.append(np.median(calls) / x)
            y_40.append(np.median(calls) / x_60)
            y_60.append(np.median(calls) / x_40)
            y_delta.append(np.median(calls_delta) / x)
            y_delta_40.append(np.median(calls_delta) / x_60)
            y_delta_60.append(np.median(calls_delta) / x_40)

            b, sn, cs, dr = 1, 0.01, 26, 0.5
            noise_levels = sorted(data[dataset][noise].keys())
            if noise in ['bayesian']:
                noise_levels.reverse()
            for nn, noise_level in enumerate(noise_levels):
                if noise == 'bayesian':
                    b = noise_level
                elif noise == 'smoothing':
                    sn = noise_level
                elif noise == 'cropping':
                    cs = noise_level
                elif noise == 'dropout':
                    dr = noise_level
                exp_name = f'{dataset}_psj_r_1_sn_{sn}_cs_{cs}_dr_{dr}_dm_l2_b_{b}_{noise}_fp_0.00_ns_{n_images}'
                raw = read_dump(exp_name)
                x = np.median(raw['model_calls'][-1, :])
                x_40 = np.percentile(raw['model_calls'][-1, :], 40)
                x_60 = np.percentile(raw['model_calls'][-1, :], 60)
                rep = data[dataset][noise][noise_level][0]
                exp_name = f'{dataset}_hsj_rep_r_{rep}_sn_{sn}_cs_{cs}_dr_{dr}_dm_l2_b_{b}_{noise}_fp_0.00_ns_{n_images}'
                raw = read_dump(exp_name)
                calls = raw['model_calls'][-1, :]
                rep = data[dataset][noise][noise_level][1]
                exp_name = f'{dataset}_hsj_rep_del_r_{rep}_sn_{sn}_cs_{cs}_dr_{dr}_dm_l2_b_{b}_{noise}_fp_0.00_ns_{n_images}'
                raw = read_dump(exp_name)
                calls_delta = raw['model_calls'][-1, :]

                z.append(1.0 / noise_level)
                y.append(np.median(calls) / x)
                y_40.append(np.percentile(calls, 50) / x_60)
                y_60.append(np.percentile(calls, 50) / x_40)
                y_delta.append(np.median(calls_delta) / x)
                y_delta_40.append(np.percentile(calls_delta, 50) / x_60)
                y_delta_60.append(np.percentile(calls_delta, 50) / x_40)

            plt.plot(z, y, label='HSJr', marker='^')
            plt.fill_between(z, y_40, y_60, alpha=0.2)
            plt.plot(z, y_delta, label='HSJr (with PSJ\'s $\\delta$)', marker='o')
            plt.fill_between(z, y_delta_40, y_delta_60, alpha=0.2)
        plt.legend()
        plt.text(0.50, 0.95, dataset.upper(), props, transform=fig.axes[0].transAxes, fontsize=28)
        plt.xlim(0, 0.5)
        plt.text(0, -0.13, 'no noise', props, transform=fig.axes[0].transAxes, fontsize=26)
        plt.text(0.93, -0.13, 'high noise', props, transform=fig.axes[0].transAxes, fontsize=26)
        # ax = fig.axes[0]
        # ax.set_xticks([0, 0.25, 0.5])  # values
        # ax.set_xticklabels(['0 \n(no noise)', '0.25', '0.5\n(high noise)'])  # labels
        plt.xlabel('temperature (T)', fontsize=30)
        plt.ylabel('relative model calls', fontsize=30)
        plt.grid()
        plt.yscale('log')
        # plt.xscale('log')
        plt.savefig(f'thesis/plots_paper/psj-vs-repeated-hsj_{dataset}_v2.pdf', bbox_inches='tight')


def adv_risk():
    datasets = ['mnist', 'cifar10']
    rc('text', usetex=True)
    rc('text.latex', preamble=[r'\usepackage{amsfonts}'])
    rc('font', size=12)  # default: 10 -> choose size depending on figsize
    rc('font', family='STIXGeneral')
    rc('legend', fontsize=12)
    plt.tight_layout(h_pad=0, w_pad=0)
    noises = ['bayesian', 'dropout', 'smoothing', 'cropping']
    noise_names_1 = ['logit', 'dropout', 'smoothing', 'cropping']
    attacks = ['hsj', 'psj']
    line_style = [(0, (5, 8)), (0, ())]
    plt.figure(figsize=(10, 4))
    ax0 = plt.subplot(1, 2, 1)
    ax1 = plt.subplot(1, 2, 2, sharey=ax0)
    ax = [ax0, ax1]
    for dd, dataset in enumerate(datasets):
        for nn, noise in enumerate(noises):
            b, sn, dr, cs = 1, 0.01, 0.5, 26
            if noise == 'cropping':
                cs = 25 if dataset == 'mnist' else 30
            if noise == 'dropout':
                dr = 0.5 if dataset == 'mnist' else 0.03
            if noise == 'smoothing':
                sn = 0.01 if dataset == 'mnist' else 0.001
            for aa, attack in enumerate(attacks):
                exp_name = f'{dataset}_{attack}_r_1_sn_{sn}_cs_{cs}_dr_{dr}_dm_l2_b_{b}_{noise}_fp_0.00_ns_100'
                raw = read_dump(exp_name, aa=True)
                AA = raw['adv_acc']
                adv_acc = AA[:, -1, :]
                adv_acc = torch.cat([adv_acc[:, :57], adv_acc[:, 58:]], dim=1)
                mean_adv_acc = torch.mean(adv_acc, dim=1)
                eps = torch.linspace(0, 10, 100)
                label = f'{attack}({noise_names_1[nn]})'
                if dataset == 'cifar10':
                    ax[dd].plot(eps[:50], mean_adv_acc[:50], linestyle=line_style[aa], color=f'C{nn + 1}', label=label)
                else:
                    ax[dd].plot(eps, mean_adv_acc, linestyle=line_style[aa], color=f'C{nn + 1}', label=label)
        ax[dd].set_xlabel('$\\epsilon$ (perturbation in unit $l_2$-norm)')
        ax[dd].grid()
        ax[dd].set_ylim(0, 1.1)
        ax[dd].text(0.5, 0.95, dataset.upper(), props, transform=ax[dd].transAxes, fontsize=15)
        if dd==0:
            # ax[dd].legend()
            ax[dd].legend(loc='upper center', bbox_to_anchor=(1.1, 1.22), ncol=4, fancybox=True, handletextpad=2.65)
        ax[dd].set_ylabel('adversarial accuracy')
    # plt.setp(ax1.get_yticklabels(), visible=False)
    image_path = f'thesis/plots_paper/adv_acc.pdf'
    plt.savefig(image_path, bbox_inches='tight')


def grad_evals(dataset='cifar10'):
    n_images = 100
    image_path = f'thesis/plots_paper/grad_evals_{dataset}_{n_images}.pdf'
    efs = [1, 2, 3, 4, 5]
    rc('text', usetex=True)
    rc('text.latex', preamble=[r'\usepackage{amsfonts}'])
    rc('font', size=14)  # default: 10 -> choose size depending on figsize
    rc('font', family='STIXGeneral')
    rc('legend', fontsize=14)
    plt.tight_layout(h_pad=0, w_pad=0)

    plt.figure(figsize=(10, 4))
    ax0 = plt.subplot(1, 2, 1)
    ax1 = plt.subplot(1, 2, 2, sharey=ax0)
    metric = 'border_distance'

    for ef in efs:
        exp_name = f'{dataset}_hsj_r_1_ef_{ef}_sn_0.01_cs_26_dr_0.5_dm_l2_b_1_deterministic_fp_0.00_ns_{n_images}'
        raw = read_dump(exp_name)
        D = raw[metric]
        C = raw['model_calls']
        if dataset == 'cifar10':
            D = torch.cat([D[:, :57], D[:, 58:]], dim=1)
            C = torch.cat([C[:, :57], C[:, 58:]], dim=1)
        dist = np.median(D, axis=1)
        calls = np.median(C, axis=1)
        # perc_40 = np.percentile(raw[metric], 40, axis=1)
        # perc_60 = np.percentile(raw[metric], 60, axis=1)
        ax0.plot(dist, label=f'factor={ef}')
        ax1.plot(calls, dist, label=f'factor={ef}')
    exp_name = f'{dataset}_hsj_true_grad_r_1_ef_1_sn_0.01_cs_26_dr_0.5_dm_l2_b_1_deterministic_fp_0.00_ns_{n_images}'
    raw = read_dump(exp_name)
    D = raw[metric]
    if dataset == 'cifar10':
        D = torch.cat([D[:, :57], D[:, 58:]], dim=1)
    dist = np.median(D, axis=1)
    # perc_40 = np.percentile(raw[metric], 40, axis=1)
    # perc_60 = np.percentile(raw[metric], 60, axis=1)
    ax0.plot(dist, label=f'true grad')
    ax0.legend()
    ax0.set_xlabel('iterations')
    ax0.set_ylabel('border distance')
    ax0.grid()
    ax1.legend()
    ax1.set_xlabel('median model calls')
    # ax1.set_ylabel('border distance')
    ax1.set_xscale('log')
    ax1.grid()
    plt.savefig(image_path, bbox_inches='tight')


# grid()
# noise()
# fig3_2lines(line=None)
# fig3_2lines(line=1)
# fig3_2lines(line=2)
# delta(dataset='mnist')
# delta(dataset='cifar10')
# hsj_vs_psj()
fig5()
# adv_risk()
# grad_evals(dataset='mnist')
# grad_evals(dataset='cifar10')

pass
