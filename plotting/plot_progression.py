import pickle
import numpy as np
import matplotlib.pylab as plt
from model_factory import get_model

TITLE = 'Infomax Progression'
NUM_ITERATIONS = 32
NUM_IMAGES = 50
TARGET_IMAGE = 3
NOISE = 'bayesian'
exp_name = 'our_on_prob_model'
image_path = 'adv/our_on_prob_model_prog.pdf'


model = get_model(key='mnist_noman', dataset='mnist')


def read_dump(path):
    raws = []
    filepath = 'adv/{}/raw_data.pkl'.format(path)
    raws.append(pickle.load(open(filepath, 'rb')))
    return raws


raws = read_dump(exp_name)


fig, ax1 = plt.subplots(figsize=(15, 7))
ax1.set_xlabel('Iterations of the Attack')
ax1.set_ylabel('Probability of True Label')
for i, raw in enumerate(raws):
    opposite = np.zeros((NUM_ITERATIONS+1, NUM_IMAGES))
    binary = np.zeros((NUM_ITERATIONS+1, NUM_IMAGES))
    approx = np.zeros((NUM_ITERATIONS+1, NUM_IMAGES))
    medians = []

    for iteration in range(NUM_ITERATIONS+1):
        distances = []
        for image in range(NUM_IMAGES):
            if image is not TARGET_IMAGE:
                continue
            if 'iterations' not in raw[image]:
                continue
            label = raw[image]['true_label']
            details = raw[image]['progression'][iteration]
            if iteration is 0:
                opp_probs = None
            else:
                opp_image = details['opposite']
                opp_probs = model.get_probs([opp_image])[0][label]

            bin_image = details['binary']
            app_image = details['approx_grad']
            app_probs = model.get_probs([app_image])[0][label]
            bin_probs = model.get_probs([bin_image])[0][label]

            approx[iteration][image] = app_probs
            opposite[iteration][image] = opp_probs
            binary[iteration][image] = bin_probs

    ax1.plot([], [], color='black', label='Grad Step')
    ax1.plot([], [], color='green', label='Opposite Movement')
    ax1.plot([], [], color='red', label='Binary Search')
    ax1.plot([0, NUM_ITERATIONS], [0.5, 0.5], '--', color='brown', label='Decision Boundary')
    for image in range(NUM_IMAGES):
        if image is not TARGET_IMAGE:
            continue
        _approx = approx[:, image]
        _binary = binary[:, image]

        _opposite = opposite[:, image]
        x, y = [0.33, 0.67], [_approx[0], _binary[0]]
        ax1.plot(x, y, color='red')
        for i in range(1, NUM_ITERATIONS+1):
            x, y = [i-0.33, i], [_binary[i-1], _approx[i]]
            ax1.plot(x, y, color='black')
            x, y = [i, i+0.33], [_approx[i], _opposite[i]]
            ax1.plot(x, y, color='green')
            x, y = [i+0.33, i+0.67], [_opposite[i], _binary[i]]
            ax1.plot(x, y, color='red')

        # x, y = [0, 0.5], [_approx[0], _binary[0]]
        # ax1.plot(x, y, color='red')
        # for i in range(1, NUM_ITERATIONS+1):
        #     x, y = [i-0.5, i], [_binary[i-1], _approx[i]]
        #     ax1.plot(x, y, color='black')
        #     x, y = [i, i+0.5], [_approx[i], _binary[i]]
        #     ax1.plot(x, y, color='red')

ax1.grid()
# plt.title(TITLE)
ax1.legend()
plt.savefig(image_path)
pass