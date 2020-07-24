import pickle
import numpy as np
import matplotlib.pylab as plt
from model_factory import get_model

NUM_ITERATIONS = 30
NUM_IMAGES = 1000

model = get_model(key='mnist_noman', dataset='mnist')


def read_dumps():
    folder = 'det_30_1000'
    filepath = 'adv/{}/raw_data.pkl'.format(folder)
    return pickle.load(open(filepath, 'rb'))


raw = read_dumps()

fig, ax1 = plt.subplots(figsize=(7, 7))
ax1.set_xlabel('Iterations of the Attack')
ax1.set_ylabel('Cosine Value')
# ax1.set_yscale('log')

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
ax1.plot(range(1, 31), medians_line, label='True Grad vs (Xt-X*)')
ax1.plot(range(1, 31), medians_estm, label='True Grad vs Estimated Grad')

ax1.grid()
plt.title('Cosine Analysis (1000 test images)')
ax1.legend()
plt.savefig('adv/cos_exp_1000.pdf')
pass
