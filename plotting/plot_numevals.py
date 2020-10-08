import pickle
import numpy as np
import matplotlib.pylab as plt
from model_factory import get_model

MAX_EVALS = 50000
NUM_ITERATIONS = 32
NUM_IMAGES = 1
# TARGET_IMAGE = 4
NOISE = 'bayesian'
exp_name = 'eval_exp'
# lis = ['infomax_5_32_opp', 'infomax_5_32_evals','infomax_5_32_opp_evals']
lis = ['eval_exp_wo_step']
image_path = 'adv/eval_exp_wo_step_bt.png'


model = get_model(key='mnist_noman', dataset='mnist')


def read_dump(path):
    filepath = 'adv/{}/raw_data.pkl'.format(path)
    raw = pickle.load(open(filepath, 'rb'))
    return raw


# raws = [read_dump(exp_name)]
raws = [read_dump(s) for s in lis]


fig, ax1 = plt.subplots(figsize=(15, 7))
ax1.set_xlabel('Iterations of the Attack')
ax1.set_ylabel('Approximate B_t')

for i, raw in enumerate(raws):
    for image in range(NUM_IMAGES):
        # if image != TARGET_IMAGE:
        #     continue
        if 'iterations' not in raw[image]:
            continue
        details = raw[image]['grad_num_evals']
        clipped = np.minimum(details, MAX_EVALS)
        ax1.plot(details, label='Image {}'.format(image))
        # ax1.plot(details, color='gray', label='Estimated Bt')
        # ax1.plot(clipped, color="black", label='Clipped Bt')
        ax1.plot([MAX_EVALS]*len(details), '--')
ax1.grid()
plt.title('Num Samples required for Grad Estimate')
ax1.legend()
plt.savefig(image_path)
pass