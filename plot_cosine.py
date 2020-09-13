import pickle
import numpy as np
import matplotlib.pylab as plt
from model_factory import get_model

NUM_ITERATIONS = 25
NUM_IMAGES = 4

model = get_model(key='mnist_noman', dataset='mnist')

G = np.round(10 ** np.linspace(0, -2, num=9), 2)


# G = G[:4]


def read_dump(path):
    raws = []
    filepath = 'adv/{}/raw_data.pkl'.format(path)
    raws.append(pickle.load(open(filepath, 'rb')))
    return raws


def read_dumps():
    raws = []
    for gamma in G:
        folder = 'det_cosv2_30_1000_g{}'.format(gamma)
        filepath = 'adv/{}/raw_data.pkl'.format(folder)
        raws.append(pickle.load(open(filepath, 'rb')))
    return raws


# raws = read_dumps()
raws = read_dump('infomax_04_25')

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 13))
fig.suptitle('Cosine Analysis - Deterministic Model (100 test images)'
             '\ntheta = g / sqrt(d)'
             '\ndelta = theta * sqrt(d) * L2_dist(t)')
ax1.set_xlabel('Iterations of the Attack')
ax1.set_ylabel('Cosine Value')
ax1.title.set_text('True Gradient vs (Xt-X*)')
ax2.set_xlabel('Iterations of the Attack')
ax2.set_ylabel('Cosine Value')
ax2.title.set_text('True Gradient vs Estimated Gradient')

data_lines = []
data_estms = []
for r, raw in enumerate(raws):
    medians_line = []
    medians_estm = []
    for iteration in range(NUM_ITERATIONS):
        distances = []
        wrt_line = []
        wrt_estm = []
        for image in range(NUM_IMAGES):
            if 'iterations' not in raw[image]:
                continue
            cos_details = raw[image]['cosine_details'][iteration]
            wrt_line.append(cos_details['true_vs_line'])
            wrt_estm.append(cos_details['true_vs_estm'])
        medians_line.append(np.median(wrt_line))
        medians_estm.append(np.median(wrt_estm))
    data_lines.append(medians_line)
    data_estms.append(medians_estm)
    ax1.plot(range(1, NUM_ITERATIONS+1), medians_line, label='g={:.2f}'.format(G[r]))
    ax2.plot(range(1, NUM_ITERATIONS+1), medians_estm, label='g={:.2f}'.format(G[r]))

np.save("cos_true_wrt_line.npy", np.array(data_lines))
np.save("cos_true_wrt_estimated.npy", np.array(data_estms))

ax1.grid()
ax2.grid()
# ax1.legend()
# ax2.legend()
plt.savefig('adv/004_bay_info_adv_cos.pdf')
pass
