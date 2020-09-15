import pickle
import numpy as np
import matplotlib.pylab as plt
from model_factory import get_model


NUM_ITERATIONS = 32
NUM_IMAGES = 1
NOISE = 'bayesian'
exp_name = 'del_later'
image_path = 'adv/del_tmap_diff.png'


model = get_model(key='mnist_noman', dataset='mnist')


def read_dump(path):
    raws = []
    filepath = 'adv/{}/raw_data.pkl'.format(path)
    raws.append(pickle.load(open(filepath, 'rb')))
    return raws


raws = read_dump(exp_name)


fig, ax1 = plt.subplots(figsize=(15, 7))
ax1.set_xlabel('Iterations of the Attack')
ax1.set_ylabel('T_map')
for i, raw in enumerate(raws):
    for iteration in range(15,16):
        for image in range(NUM_IMAGES):
            if 'iterations' not in raw[image]:
                continue
            details = raw[image]['progression'][iteration]
            tmap = details['tmap']
            smoothing_kernel = np.ones(10,)/10
            diffs = np.abs(np.diff(tmap))
            smoothed_diffs = np.convolve(diffs, smoothing_kernel, mode='same')
            ax1.plot(diffs[1500:2000], label="diffs")
            ax1.plot(smoothed_diffs[1500:2000])
ax1.grid()
plt.title('Infomax Progression')
ax1.legend()
plt.savefig(image_path)
pass