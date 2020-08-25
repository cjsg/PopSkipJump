import pickle
import numpy as np
import matplotlib.pylab as plt
from model_factory import get_model


NUM_ITERATIONS = 32
NUM_IMAGES = 1
NOISE = 'bayesian'
exp_name = 'infomax_1_32_opp'
image_path = 'adv/del_later.png'


model = get_model(key='mnist_noman', dataset='mnist')


def read_dump(path):
    raws = []
    filepath = 'adv/{}/raw_data.pkl'.format(path)
    raws.append(pickle.load(open(filepath, 'rb')))
    return raws


raws = read_dump(exp_name)


fig, ax1 = plt.subplots(figsize=(8, 7))
ax1.set_xlabel('Iterations of the Attack')
ax1.set_ylabel('Model Calls')
for i, raw in enumerate(raws):
    calls = np.zeros((NUM_ITERATIONS+1, NUM_IMAGES))
    for iteration in range(NUM_ITERATIONS):
        for image in range(NUM_IMAGES):
            if 'iterations' not in raw[image]:
                continue
            calls[0][image] = raw[image]['model_calls']['projection']
            details = raw[image]['model_calls']['iters']
            calls[iteration+1][image] = details[iteration]['binary']
    total_calls = calls.sum(axis=1)
    ax1.plot(total_calls)
ax1.grid()
plt.title('Model Calls Analysis')
ax1.legend()
plt.savefig(image_path)
pass