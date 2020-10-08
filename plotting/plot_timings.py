import pickle
import torch
import numpy as np
import matplotlib.pylab as plt
from model_factory import get_model
from tracker import Diary

NUM_ITERATIONS = 32
NUM_IMAGES = 100
# TARGET_IMAGE = 4
NOISE = 'bayesian'
exp_name = 'eval_exp_wo_step_50'
# lis = ['infomax_5_32_opp', 'infomax_5_32_evals','infomax_5_32_opp_evals']
# lis = ['gpu_cpu', 'gpu', 'gpu_exp', 'gpu_approxgrad', 'gpu_bernoulli', 'gpu_cj', 'gpu_queries_20']
# titles = ['CPU', 'GPU', 'GPU (in batches)', 'Approximate Gradient on GPU now',
#           'Bernoulli instead of model calls inside bin_search',
#           'CJ Optimization (Interval)', 'CJ Optimization (Queries=5)']
lis = ['gpu', 'gpu_pf', 'gpu_pf_q', 'gpu_pf_q_gq']
titles = ['PF=1.0, Q=1, GQ=1', 'PF=0.1, Q=1, GQ=1', 'PF=0.1, Q=5, GQ=1', 'PF=0.1, Q=5, GQ=5']
# lis = lis[-2:]
# titles = titles[-2:]
image_path = 'adv/timings.pdf'

model = get_model(key='mnist_noman', dataset='mnist')


def read_dump(path):
    filepath = 'adv/{}/raw_data.pkl'.format(path)
    try:
        raw = torch.load(open(filepath, 'rb'), map_location='cpu')
    except:
        raw = pickle.load(open(filepath, 'rb'))
    return raw


# raws = [read_dump(exp_name)]
raws = [read_dump(s) for s in lis]

fig = plt.figure(figsize=(20, 5 * len(lis)))
for i, raw in enumerate(raws):
    t_approx_grad, t_step_search, t_bin_search, t_total = 0, 0, 0, 0
    t_num_evals = 0
    for iteration in range(NUM_ITERATIONS):
        for image in range(NUM_IMAGES):
            diary: Diary = raw[image]
            if len(diary.iterations) is 0:
                continue
            epoch = diary.epoch_start
            init_search = diary.epoch_initial_bin_search - epoch
            details = raw[image].iterations[iteration].time
            start = details.start
            # num_evals = details[iteration]['num_evals']
            approx_grad = details.approx_grad
            step_search = details.step_search
            bin_search = details.bin_search
            end = details.end

            # t_num_evals += num_evals - start
            # t_approx_grad += approx_grad - num_evals
            t_approx_grad += approx_grad - start
            t_step_search += step_search - approx_grad
            t_bin_search += bin_search - step_search
            t_total += end - start
    t_approx_grad /= NUM_IMAGES
    t_bin_search /= NUM_IMAGES
    t_total /= NUM_IMAGES
    labels = ['Approx Grad', 'Binary Search']
    values = [t_approx_grad, t_bin_search]
    print(values, t_total)
    plt.subplot(len(raws) / 2 + 1, 2, i + 1)
    plt.pie(values, labels=labels)
    # plt.text(-2, 1.1, "Num Evals: {} secs".format(np.round(t_num_evals, 1)))
    plt.text(-2, 0.9, "Approx Grad: {} secs/image".format(np.round(t_approx_grad, 1)))
    # plt.text(-2, 0.9, "Step Search: {} secs".format(np.round(t_step_search, 1)))
    plt.text(-2, 0.8, "Binary Search: {} secs/image".format(np.round(t_bin_search, 1)))
    plt.text(-2, 0.7, "Total: {} secs/image".format(np.round(t_total, 1)))
    plt.title(titles[i])
    # plt.bar(['Approx Grad'], [t_approx_grad])
    # plt.bar(['Step Search'], [t_step_search])
    # plt.bar(['Binary Search'], [t_bin_search])
    # plt.bar(['Total'], [t_total])
    # ax1.plot(range(1, NUM_ITERATIONS+2), total_bin_calls, label='Total', color='grey')
plt.suptitle(f"Timing Analysis for {NUM_IMAGES} images\n"
             f"PF: prior fraction in infomax\n"
             f"Q: queries per point in infomax\n"
             f"GQ: queries per point in approx. gradient")
# plt.ylabel('Time (in seconds)')
plt.savefig(image_path)
pass
