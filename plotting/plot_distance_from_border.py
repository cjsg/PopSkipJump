import sys
import numpy as np
import torch
import matplotlib.pylab as plt
from tqdm import tqdm
from model_factory import get_model
from img_utils import get_device
from tracker import Diary

OUT_DIR = 'aistats'
NUM_ITERATIONS = 32
NUM_IMAGES = 100
eps = list(range(1, 6))


def read_dump(path):
    filepath = f'{OUT_DIR}/{path}/raw_data.pkl'
    # raw = torch.load(open(filepath, 'rb'), map_location='cpu')
    raw = torch.load(open(filepath, 'rb'))
    return raw


exps = {
    'bayesian': 'all_bay',
    'deterministic': 'all_det',
    'stochastic': 'all_sto',
    'temperature': 'bay_temp'
}

dumps = {
    'deterministic': [('hsj_b_1_deterministic_ns_100', 'hsj'),
                      ('hsj_rep_b_1_deterministic_ns_100', 'hsjr'),
                      ('psj_b_1_deterministic_ns_100', 'psj')],
    'stochastic': [('hsj_b_1_stochastic_ns_100', 'hsj'),
                   ('hsj_rep_b_1_stochastic_ns_100', 'hsjr'),
                   ('psj_b_1_stochastic_ns_100', 'psj')],
    'bayesian': [('hsj_b_1_bayesian_ns_100', 'hsj'),
                 ('hsj_rep_b_1_bayesian_ns_100', 'hsjr'),
                 ('psj_b_1_bayesian_ns_100', 'psj')],
    'temperature': [('psj_b_1_bayesian_ns_100', 'beta=1'),
                    ('psj_b_5_bayesian_ns_100', 'beta=5'),
                    ('psj_b_10_bayesian_ns_100', 'beta=10')],
}

noise = sys.argv[1]
exp = exps[noise]
flip_prob = 0.05
theta_det = 1 / (28 * 28 * 28)
beta = 1

# raws = [read_dump(f'{attack}_{rep}_b_{beta}_bayesian_ns_5') for beta in betas]
raws = [read_dump(s) for (s, _) in dumps[noise]]
device = get_device()
model = get_model(key='mnist_noman', dataset='mnist')
model.model = model.model.to(device)


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


D = torch.zeros(size=(len(raws), NUM_ITERATIONS, NUM_IMAGES), device=device)
AA = torch.zeros(size=(len(raws), len(eps), NUM_ITERATIONS, NUM_IMAGES), device=device)
MC = torch.zeros_like(D, device=device)
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

            D[i, iteration, image] = torch.norm(x_star - x_tt) ** 2 / 784 / 1 ** 2
            MC[i, iteration, image] = calls
            for j in range(len(eps)):
                x_adv = x_star + eps[j] * (x_tt - x_star) / torch.norm(x_tt - x_star)
                p_adv = model.get_probs(x_adv[None])[0]
                if noise == 'bayesian':
                    AA[i, j, iteration, image] = p_adv[label]
                elif noise == 'deterministic':
                    AA[i, j, iteration, image] = (torch.argmax(p_adv) == label) * 1.0
                else:
                    p_temp = torch.ones_like(p_adv) * flip_prob / (p_adv.shape[0] - 1)
                    pred = torch.argmax(p_adv)
                    p_temp[pred] = 1 - flip_prob
                    AA[i, j, iteration, image] = p_temp[label]

plt.figure(figsize=(10, 7))
image_path = f'{OUT_DIR}/{exp}_distance'
for i in range(len(raws)):
    plt.plot(np.median(D[i], axis=1), label=dumps[noise][i][1])
plt.legend()
plt.grid()
plt.savefig(f'{image_path}.png')
plt.savefig(f'{image_path}.pdf')

for j in range(len(eps)):
    plt.figure(figsize=(10, 7))
    image_path = f'{OUT_DIR}/{exp}_risk'
    for i in range(len(raws)):
        plt.plot(np.mean(AA[i, j], axis=1), label=dumps[noise][i][1])
    plt.legend()
    plt.grid()
    plt.savefig(f'{image_path}_{eps[j]}.png')
    plt.savefig(f'{image_path}_{eps[j]}.pdf')

plt.figure(figsize=(10, 7))
image_path = f'{OUT_DIR}/{exp}_calls'
for i in range(len(raws)):
    plt.plot(np.mean(MC[i], axis=1), label=dumps[noise][i][1])
plt.legend()
plt.grid()
plt.yscale('log')
plt.savefig(f'{image_path}.png')
plt.savefig(f'{image_path}.pdf')
