import pickle
import numpy as np
import matplotlib.pylab as plt
from model_factory import get_model


NUM_ITERATIONS = 30
NUM_IMAGES = 3
FF = [1, 32]
FP = [0.5, 0.8]
NOISE = 'bayesian'


FLIP_PROB = 0.2

model = get_model(key='mnist_noman', dataset='mnist')


def read_dump(path):
    raws = []
    filepath = 'adv/{}/raw_data.pkl'.format(path)
    raws.append(pickle.load(open(filepath, 'rb')))
    return raws


def read_dumps(noise='bayesian'):
    raws = []
    if noise == 'stochastic':
        for flip in FP:
            folder = 'sto_{}_{}_{}'.format(NUM_ITERATIONS, flip, 32)
            filepath = 'adv/{}/raw_data.pkl'.format(folder)
            raws.append(pickle.load(open(filepath, 'rb')))
    elif noise == 'bayesian':
        SLACK = 0.10
        for freq in FF:
            folder = 'approxgrad_{}_gsf{}_sf{}_avg'.format(NUM_ITERATIONS, freq, 32)
            filepath = 'adv/{}/raw_data.pkl'.format(folder)
            raws.append(pickle.load(open(filepath, 'rb')))
    else:
        raise RuntimeError
    return raws


# raws = read_dumps(noise=NOISE)
raws = read_dump('del_later')


fig, ax1 = plt.subplots(figsize=(7, 7))
ax1.set_xlabel('Iterations of the Attack')
ax1.set_ylabel('L2 Distance')
ax2 = ax1.twinx()
ax2.set_ylabel('Proportion of Non-adversarial samples')

for i, raw in enumerate(raws):
    adv_count = np.zeros(NUM_ITERATIONS, )
    medians = []
    for iteration in range(NUM_ITERATIONS):
        distances = []
        for image in range(NUM_IMAGES):
            if 'iterations' not in raw[image]:
                continue
            distance = raw[image]['iterations'][iteration]['distance']
            adversarial = raw[image]['iterations'][iteration]['perturbed']
            if NOISE == 'stochastic':
                prediction = model.ask_model([adversarial])
                if prediction[0] != raw[image]['true_label']:
                    adv_count[iteration] += 1
            elif NOISE == 'bayesian':
                true_slack = 0.0
                probs = model.get_probs([adversarial])
                label = raw[image]['true_label']
                adv_prob = np.max(probs[0][np.arange(10) != label])
                if adv_prob > probs[0][label] + true_slack:
                    adv_count[iteration] += 1
            else:
                raise RuntimeError
            distances.append(distance)
        medians.append(np.median(distances))
    if NOISE == 'stochastic':
        ax1.plot(medians, label='flip_prob = {}'.format(FP[i]))
        ax2.plot(1-adv_count/NUM_IMAGES, '--', label='flip_prob = {}'.format(FP[i]))
    if NOISE == 'bayesian':
        ax1.plot(medians, label='C = {}'.format(FF[i]))
        ax2.plot(1-adv_count/NUM_IMAGES, '--', label='C = {}'.format(FF[i]))

ax1.grid()
# plt.ylabel('L2 Distance')
# plt.ylabel('Number of adversarial using true logits')
# plt.xlabel('Iterations of Attack')
plt.title('Using Estimate #2')
ax1.legend()
plt.savefig('adv/del_later2.png'.format(NOISE))
pass