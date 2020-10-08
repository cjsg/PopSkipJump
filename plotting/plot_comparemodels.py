import torch
import numpy as np
import matplotlib.pylab as plt
from tracker import Diary
from model_factory import get_model

dumps = ['gpu', 'gpu_pf', 'gpu_pf_q', 'gpu_pf_q_gq']
labels = ['PF=1.0, Q=1, GQ=1', 'PF=0.1, Q=1, GQ=1', 'PF=0.1, Q=5, GQ=1', 'PF=0.1, Q=5, GQ=5']
NUM_ITERATIONS = 32
image_path = 'adv/del_later.pdf'


def read_dump(name):
    filepath = 'adv/{}/raw_data.pkl'.format(name)
    return torch.load(open(filepath, 'rb'), map_location='cpu')


raws = [read_dump(s) for s in dumps]
model = get_model(key='mnist_noman', dataset='mnist')

fig, ax1 = plt.subplots(figsize=(7, 7))
for r, raw in enumerate(raws):
    n_imgs = len(raw)
    dist = np.zeros(shape=(NUM_ITERATIONS, n_imgs))
    prob = np.zeros(shape=(NUM_ITERATIONS, n_imgs))
    for iteration in range(NUM_ITERATIONS):
        for image in range(n_imgs):
            diary: Diary = raw[image]
            x_t = diary.iterations[iteration].bin_search.numpy()
            x_star = diary.original.numpy()
            label = diary.true_label
            p_t = model.get_probs([x_t])[0][label]
            d_t = np.linalg.norm(x_t - x_star) ** 2 / (28*28)
            dist[iteration, image] = d_t
            prob[iteration, image] = p_t
    plot_dist = np.median(dist, axis=1)
    plot_prob = np.mean(prob, axis=1)
    ax1.scatter(plot_dist[-3:], plot_prob[-3:], label=labels[r])
ax1.set_xlabel('median l2 distance')
ax1.set_ylabel('mean prob of true label')
ax1.grid()
ax1.legend()
plt.savefig(image_path)
