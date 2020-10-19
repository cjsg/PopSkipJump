import os
import sys
import numpy as np
import torch
import matplotlib.pylab as plt
from matplotlib import rc, rcParams
from tqdm import tqdm
from model_factory import get_model
from img_utils import get_device
from tracker import Diary

OUT_DIR = 'aistats'
device = get_device()
NUM_ITERATIONS = 32
NUM_IMAGES = 1
eps = list(range(1, 6))


def read_dump(path):
    filepath = f'{OUT_DIR}/{path}/raw_data.pkl'
    raw = torch.load(open(filepath, 'rb'), map_location=device)
    # raw = torch.load(open(filepath, 'rb'))
    return raw


exps = {
    'bayesian': 'all_bay',
    'deterministic': 'all_det',
    'stochastic': 'all_sto',
    'stochastic_0.10': 'all_sto_0.10',
    'temperature': 'psj_temp',
    'delta': 'hsj_delta',
    'dummy': 'dummy'
}

dumps = {
    'deterministic': [('hsj_b_1_deterministic_ns_100', 'HSJ'),
                      ('hsj_rep_b_1_deterministic_ns_100', 'HSJ-r'),
                      ('psj_b_1_deterministic_ns_100', 'PSJ')],
    'stochastic': [('hsj_b_1_stochastic_ns_100', 'HSJ'),
                   ('hsj_rep_b_1_stochastic_ns_100', 'HSJ-r'),
                   ('psj_b_1_stochastic_ns_100', 'PSJ')],
    'stochastic_0.10': [('hsj_b_1_stochastic_ns_100_fp_0.10', 'HSJ'),
                        ('hsj_rep_b_1_stochastic_ns_100_fp_0.10', 'HSJ-r'),
                        ('psj_b_1_stochastic_ns_100_fp_0.10', 'PSJ')],
    'bayesian': [('hsj_b_1_bayesian_ns_100', 'HSJ'),
                 ('hsj_rep_b_1_bayesian_ns_100', 'HSJ-r'),
                 ('psj_b_1_bayesian_ns_100', 'PSJ')],
    'delta': [('hsj_rep_psj_delta_b_1_bayesian_ns_100', "HSJ's delta (bayesian)"),
              ('hsj_rep_psj_delta_b_1_deterministic_ns_100', "HSJ's delta (deterministic)"),
              ('hsj_rep_b_1_bayesian_ns_100', "PSJ's delta (bayesian)"),
              ('hsj_rep_b_1_deterministic_ns_100', "PSJ's delta (deterministic)")],
    'temperature': [('psj_b_1_bayesian_ns_100', 'PSJ(beta=1)'),
                    ('psj_b_5_bayesian_ns_100', 'PSJ(beta=5)'),
                    ('psj_b_50_bayesian_ns_100', 'PSJ(beta=50)'),
                    ('psj_b_1_deterministic_ns_100', 'PSJ(deterministic)'),
                    ('hsj_b_1_deterministic_ns_100', 'HSJ(deterministic)')],
    'dummy': [('del_later_hsj', 'HSJ'),
              ('del_later_psj', 'PSJ'),
              ('del_later_psj_q', 'PSJ-Q'),
              ('del_later_psj_pf', 'PSJ-PF'),
              ('del_later', 'CUR')]
}

noise = sys.argv[1]
exp = exps[noise]
flip_prob = 0.05
beta = 1

# raws = [read_dump(f'{attack}_{rep}_b_{beta}_bayesian_ns_5') for beta in betas]
raws = [read_dump(s) for (s, _) in dumps[noise]]
dataset = 'cifar10'
if dataset == 'mnist':
    model = get_model(key='mnist_noman', dataset=dataset)
    d = 28. * 28.
else:
    model = get_model(key='cifar10', dataset=dataset)
    d = 32. * 32. * 3.
model.model = model.model.to(device)
theta_det = 1. / (np.sqrt(d) * d)


def search_boundary(x_star, x_t, theta_det, true_label):
    high, low = 1, 0
    while high - low > theta_det:
        mid = (high + low) / 2.0
        x_mid = (1 - mid) * x_star + mid * x_t
        pred = torch.argmax(model.get_probs(x_mid[None])[0])
        if pred == true_label:
            low = mid
        else:
            high = mid
    out = (1 - high) * x_star + high * x_t
    return out


def project(x_star, x_t, label, theta_det):
    probs = model.get_probs(x_t[None])
    if torch.argmax(probs[0]) == label:
        c = 0.25
        while True:
            x_tt = x_t + c * (x_t - x_star) / torch.norm(x_t - x_star)
            pred = torch.argmax(model.get_probs(x_tt[None])[0])
            if pred != label:
                x_tt = search_boundary(x_t, x_tt, theta_det, label)
                break
            c += c
    else:
        x_tt = search_boundary(x_star, x_t, theta_det, label)
    return x_tt


D = torch.zeros(size=(len(raws), NUM_ITERATIONS+1, NUM_IMAGES), device=device)
D_OUT = torch.zeros_like(D, device=device)
AA = torch.zeros(size=(len(raws), len(eps), NUM_ITERATIONS+1, NUM_IMAGES), device=device)
MC = torch.zeros_like(D, device=device)
SS = torch.zeros_like(D, device=device)
for i, raw in enumerate(raws):
    print(f"Scanning Dump {i}...")
    for iteration in tqdm(range(NUM_ITERATIONS)):
        for image in range(NUM_IMAGES):
            diary: Diary = raw[image]
            details = diary.iterations
            x_star = diary.original
            label = diary.true_label
            calls = details[iteration].calls.bin_search
            x_t = details[iteration].bin_search
            x_tt = project(x_star, x_t, label, theta_det)
            p_tt = model.get_probs(x_tt[None])[0][label]
            x_0 = project(x_star, diary.initial_projection, label, theta_det)
            D[i, 0, image] = torch.norm(x_star - x_0) ** 2 / d
            D[i, iteration+1, image] = torch.norm(x_star - x_tt) ** 2 / d / 1 ** 2
            if dumps[noise][i][0].startswith('psj'):
                d_output = D[i, iteration+1, image]
            else:
                d_output = details[iteration].distance
            # D_OUT[i, iteration+1, image] = d_output
            MC[i, iteration+1, image] = calls
            SS[i, iteration+1, image] = details[iteration].calls.step_search - details[iteration].calls.approx_grad

            # for j in range(len(eps)):
            #     x_adv = x_star + eps[j] * (x_tt - x_star) / torch.norm(x_tt - x_star)
            #     p_adv = model.get_probs(x_adv[None])[0]
            #     if noise == 'bayesian':
            #         AA[i, j, iteration+1, image] = p_adv[label]
            #     elif noise == 'deterministic':
            #         AA[i, j, iteration+1, image] = (torch.argmax(p_adv) == label) * 1.0
            #     else:
            #         p_temp = torch.ones_like(p_adv) * flip_prob / (p_adv.shape[0] - 1)
            #         pred = torch.argmax(p_adv)
            #         p_temp[pred] = 1 - flip_prob
            #         AA[i, j, iteration+1, image] = p_temp[label]

D = D.cpu().numpy()
D_OUT = D_OUT.cpu().numpy()
MC = MC.cpu().numpy()
SS = SS.cpu().numpy()
AA = AA.cpu().numpy()

PLOTS_DIR = f'{OUT_DIR}/plots_{exp}'
if not os.path.exists(PLOTS_DIR):
    os.makedirs(PLOTS_DIR)

rc('text', usetex=True)
rc('text.latex', preamble=[r'\usepackage{amsfonts}'])
rc('font', size=14)  # default: 10 -> choose size depending on figsize
rc('font', family='STIXGeneral')
rc('legend', fontsize=16)
plt.tight_layout(h_pad=0, w_pad=.5)

plt.figure(figsize=(10, 7))
image_path = f'{PLOTS_DIR}/distance'
for i in range(len(raws)):
    plt.plot(np.median(D[i], axis=1), label=dumps[noise][i][1])
plt.legend()
plt.grid()
plt.savefig(f'{image_path}.png', bbox_inches='tight', pad_inches=.02)
plt.savefig(f'{image_path}.pdf', bbox_inches='tight', pad_inches=.02)

# plt.figure(figsize=(10, 7))
# image_path = f'{PLOTS_DIR}/distance_output'
# for i in range(len(raws)):
#     plt.plot(np.median(D_OUT[i], axis=1), label=dumps[noise][i][1])
# plt.legend()
# plt.grid()
# plt.savefig(f'{image_path}.png', bbox_inches='tight', pad_inches=.02)
# plt.savefig(f'{image_path}.pdf', bbox_inches='tight', pad_inches=.02)

# for j in range(len(eps)):
#     plt.figure(figsize=(10, 7))
#     image_path = f'{PLOTS_DIR}/adv_acc'
#     for i in range(len(raws)):
#         plt.plot(np.mean(AA[i, j], axis=1), label=dumps[noise][i][1])
#     plt.legend()
#     plt.grid()
#     plt.savefig(f'{image_path}_{eps[j]}.png', bbox_inches='tight', pad_inches=.02)
#     plt.savefig(f'{image_path}_{eps[j]}.pdf', bbox_inches='tight', pad_inches=.02)

plt.figure(figsize=(10, 7))
image_path = f'{PLOTS_DIR}/calls'
for i in range(len(raws)):
    plt.plot(np.mean(SS[i], axis=1), label=dumps[noise][i][1])
plt.legend()
plt.grid()
# plt.yscale('log')
plt.savefig(f'{image_path}.png', bbox_inches='tight', pad_inches=.02)
plt.savefig(f'{image_path}.pdf', bbox_inches='tight', pad_inches=.02)
