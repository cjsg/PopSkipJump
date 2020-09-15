import pickle
import numpy as np
import matplotlib.pylab as plt
from model_factory import get_model


NUM_ITERATIONS = 32
NUM_IMAGES = 50
# TARGET_IMAGE = 4
NOISE = 'bayesian'
lis = ['our_on_prob_model', 'our_on_prob_model_beta_5', 'our_on_prob_model_beta_10',
       'our_on_prob_model_beta_50', 'our_on_prob_model_beta_100', 'beta_det_1_32', 'beta_hsja_1_32']
betas = ['beta=1', 'beta=5', 'beta: 10', 'beta: 50', 'beta: 100','deterministic', 'det_hsja']
lis = lis[:5]
betas = betas[:5]

# lis = ['eval_exp_wo_step_50']
image_path = 'adv/our_on_prob_model_beta.pdf'


model = get_model(key='mnist_noman', dataset='mnist')


def read_dump(path):
    filepath = 'adv/{}/raw_data.pkl'.format(path)
    raw = pickle.load(open(filepath, 'rb'))
    return raw


# raws = [read_dump(exp_name)]
raws = [read_dump(s) for s in lis]


fig, ax1 = plt.subplots(figsize=(15, 7))
ax1.set_xlabel('Iterations of the Attack')
ax1.set_ylabel('Mean Model Calls')
ALPHA = 0.4
ax1.plot([], [], color='black', label='Approximating Gradient', alpha=ALPHA)
ax1.plot([], [], color='blue', label='Grad Step Search', alpha=ALPHA)
ax1.plot([], [], color='red', label='Binary Search', alpha=ALPHA)
for i, raw in enumerate(raws):
    grad_calls = np.zeros((NUM_ITERATIONS+1, NUM_IMAGES))
    step_calls = np.zeros((NUM_ITERATIONS+1, NUM_IMAGES))
    bin_calls = np.zeros((NUM_ITERATIONS+1, NUM_IMAGES))
    for iteration in range(NUM_ITERATIONS):
        for image in range(NUM_IMAGES):
            # if image != TARGET_IMAGE:
            #     continue
            if 'iterations' not in raw[image]:
                continue
            grad_calls[0][image] = raw[image]['model_calls']['initialization']
            bin_calls[0][image] = raw[image]['model_calls']['projection']
            step_calls[0][image] = None

            details = raw[image]['model_calls']['iters']
            grad_calls[iteration+1][image] = details[iteration]['approx_grad']
            step_calls[iteration+1][image] = details[iteration]['step_search']
            bin_calls[iteration+1][image] = details[iteration]['binary']
    total_grad_calls = grad_calls.mean(axis=1)
    total_step_calls = step_calls.mean(axis=1)
    total_bin_calls = bin_calls.mean(axis=1)
    for iteration in range(0, NUM_ITERATIONS+1):
        if iteration is 0:
            pass
            # ax1.plot([iteration, iteration + 0.33],
            #          [0, total_grad_calls[iteration]],
            #          color="green")
        else:
            ax1.plot([iteration, iteration+0.33],
                     [total_bin_calls[iteration-1], total_grad_calls[iteration]],
                     color="black", alpha=ALPHA)
        ax1.plot([iteration+0.33, iteration+0.67],
                     [total_grad_calls[iteration], total_step_calls[iteration]],
                 color="blue", alpha=ALPHA)
        ax1.plot([iteration+0.67, iteration+1],
                 [total_step_calls[iteration], total_bin_calls[iteration]],
                 color="red", alpha=ALPHA)
    ax1.plot(range(1, NUM_ITERATIONS+2), total_bin_calls, label=betas[i])
ax1.grid()
plt.title('Model Calls Analysis (50 random images)')
ax1.legend()
plt.yscale('log')
plt.savefig(image_path)
pass