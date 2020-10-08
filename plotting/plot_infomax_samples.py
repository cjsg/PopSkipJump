import pickle
import numpy as np
import matplotlib.pylab as plt
from model_factory import get_model


NUM_ITERATIONS = 32
NUM_IMAGES = 1
# TARGET_IMAGE = 4
NOISE = 'bayesian'
exp_name = 'eval_exp_wo_step_50'
# lis = ['infomax_5_32_opp', 'infomax_5_32_evals','infomax_5_32_opp_evals']
lis = ['del_later']
image_path = 'adv/del_later.pdf'


model = get_model(key='mnist_noman', dataset='mnist')


def read_dump(path):
    filepath = 'adv/{}/raw_data.pkl'.format(path)
    raw = pickle.load(open(filepath, 'rb'))
    return raw


# raws = [read_dump(exp_name)]
raws = [read_dump(s) for s in lis]

plt.figure(figsize=(10, 25))
plt.suptitle('Infomax Samples Analysis')
for i, raw in enumerate(raws):
    t_approx_grad, t_step_search, t_bin_search, t_total = 0, 0, 0, 0
    ticks = [0, 6, 12, 18, 24, 30]
    for j, iteration in enumerate(ticks):
        for image in range(NUM_IMAGES):
            # if image != TARGET_IMAGE:
            #     continue
            if 'iterations' not in raw[image]:
                continue
            samples = raw[image]['progression'][iteration+1]['samples']
            break

        plt.subplot(len(ticks), 2, 2*j + 1)
        plt.title('Sampled Points Progression (Iteration {})'.format(iteration))
        plt.ylabel('Sampled Point (x_j)')
        plt.plot(samples)
        plt.subplot(len(ticks), 2, 2*j + 2)
        plt.title('Sampled Points Histogram (Iteration {})'.format(iteration))
        plt.ylabel('Frequency')
        plt.hist(samples, bins=100)
    # ax1.plot(range(1, NUM_ITERATIONS+2), total_bin_calls, label='Total', color='grey')
# ax1.grid()
plt.savefig(image_path)
pass