import numpy as np
import torch
import matplotlib.pylab as plt
from tqdm import tqdm
from model_factory import get_model
from tracker import Diary

OUT_DIR = 'aistats'
NUM_ITERATIONS = 32
NUM_IMAGES = 5


def read_dump(path):
    filepath = f'{OUT_DIR}/{path}/raw_data.pkl'
    raw = torch.load(open(filepath, 'rb'), map_location='cpu')
    return raw


dumps = [
            # ('hsj_b_20_deterministic_ns_100', 'hsj'),
            # ('hsj_rep_b_20_deterministic_ns_100', 'hsjr'),
            # ('psj_b_20_deterministic_ns_100', 'psj'),
            ('hsj_b_1_stochastic_ns_5', 'hsj'),
            ('hsj_rep_b_1_stochastic_ns_5', 'hsjr'),
            ('psj_b_1_stochastic_ns_5', 'psj'),
            # ('hsj_b_1_bayesian_ns_100', 'hsj'),
            # ('hsj_rep_b_1_bayesian_ns_100', 'hsjr'),
            # ('psj_b_1_bayesian_ns_100', 'psj')
        ]

theta_det = 1 / (28 * 28 * 28)
beta = 1

# raws = [read_dump(f'{attack}_{rep}_b_{beta}_bayesian_ns_5') for beta in betas]
raws = [read_dump(s) for (s,_) in dumps]
model = get_model(key='mnist_noman', dataset='mnist')


def search_boundary(x_star, x_t, theta_det, true_label):
    high, low = 1, 0
    while high - low > theta_det:
        mid = (high + low) / 2.0
        x_mid = (1 - mid) * x_star + mid * x_t
        pred = np.argmax(model.get_probs(x_mid[None])[0])
        if pred == true_label:
            low = mid
        else:
            high = mid
    out = (1 - high) * x_star + high * x_t
    return out


def project(x_star, x_t, label, theta_det):
    probs = model.get_probs([x_t]).numpy()
    if np.argmax(probs[0]) == label:
        c = 0.25
        while True:
            x_tt = x_t + c*(x_t - x_star)/np.linalg.norm(x_t - x_star)
            pred = np.argmax(model.get_probs(x_tt[None])[0])
            if pred != label:
                x_tt = search_boundary(x_t, x_tt, theta_det, label)
                break
            c += c
    else:
        x_tt = search_boundary(x_star, x_t, theta_det, label)
    return x_tt


D = np.zeros(shape=(len(raws), NUM_ITERATIONS, NUM_IMAGES))
AR = np.zeros_like(D)
MC = np.zeros_like(D)
for i, raw in enumerate(raws):
    print(f"Scanning Dump {i}...")
    for iteration in tqdm(range(NUM_ITERATIONS)):
        for image in range(NUM_IMAGES):
            diary: Diary = raw[image]
            details = diary.iterations
            x_star = diary.original.numpy()
            label = diary.true_label
            calls = details[iteration].calls.bin_search
            x_t = details[iteration].bin_search.numpy()
            x_tt = project(x_star, x_t, label, theta_det)
            p_tt = model.get_probs([x_tt]).numpy()[0][label]
            D[i, iteration, image] = np.linalg.norm(x_star - x_tt) ** 2 / 784 / 1 ** 2
            AR[i, iteration, image] = 1 - p_tt
            MC[i, iteration, image] = calls


plt.figure(figsize=(10, 7))
image_path = f'{OUT_DIR}/all_sto_distance'
for i in range(len(raws)):
    plt.plot(np.median(D[i], axis=1), label=dumps[i][1])
plt.legend()
plt.grid()
plt.savefig(f'{image_path}.png')
plt.savefig(f'{image_path}.pdf')

plt.figure(figsize=(10, 7))
image_path = f'{OUT_DIR}/all_sto_risk'
for i in range(len(raws)):
    plt.plot(np.mean(AR[i], axis=1), label=dumps[i][1])
plt.legend()
plt.grid()
plt.savefig(f'{image_path}.png')
plt.savefig(f'{image_path}.pdf')

plt.figure(figsize=(10, 7))
image_path = f'{OUT_DIR}/all_sto_calls'
for i in range(len(raws)):
    plt.plot(np.mean(MC[i], axis=1), label=dumps[i][1])
plt.legend()
plt.grid()
plt.yscale('log')
plt.savefig(f'{image_path}.png')
plt.savefig(f'{image_path}.pdf')
