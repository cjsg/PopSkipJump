import pickle
import numpy as np
import matplotlib.pylab as plt
from model_factory import get_model


NUM_ITERATIONS = 32
NUM_IMAGES = 25
NOISE = 'bayesian'
# exp_names = ['det_25', 'hsja_on_det_model_ourtheta_25', 'prob_25', 'prob_prior_25', 'our_gradstep_25',
#              'hsja_on_det_model_ourtheta_25_v2', 'det_25_v2']
exp_names = ['det_25_v2', 'hsja_on_det_model_ourtheta_25_v2', 'prob_25_deltainit_v2', 'prob_prior_25', 'our_gradstep_25']
labels = ['HSJA', 'HSJA (Our Theta)', 'Our Attack', 'Our Attack (with x_t projected to boundary)',
          'Our Attack (prior disabled)', 'Our Attack (grad step det)']
exp_names = [exp_names[i] for i in [0,1,2,4]]
labels = [labels[i] for i in [0,1,2,3,5]]

image_path = 'adv/debug_25.pdf'
ALPHA = 0.4

model = get_model(key='mnist_noman', dataset='mnist')


def read_dump(path):
    filepath = 'adv/{}/raw_data.pkl'.format(path)
    return pickle.load(open(filepath, 'rb'))


raws = [read_dump(x) for x in exp_names]
theta_det = 1 / (28*28*28)


def search_boundary(x_star, x_t, theta_det, true_label):
    high, low = 1,0
    while high - low > theta_det:
        mid = (high + low) / 2.0
        x_mid = (1-mid)*x_star + mid*x_t
        pred = np.argmax(model.get_probs(x_mid[None])[0])
        if pred == true_label:
            low = mid
        else:
            high = mid
    out = (1-high)*x_star + high*x_t
    return out


plot1series1, plot1series2,plot2series = [], [], []
for i, raw in enumerate(raws):
    abs_error, abs_error2 = np.zeros(NUM_ITERATIONS, ), np.zeros(NUM_ITERATIONS, )
    medians, medians2 = [], []
    grad_calls = np.zeros((NUM_ITERATIONS+1, NUM_IMAGES))
    step_calls = np.zeros((NUM_ITERATIONS+1, NUM_IMAGES))
    bin_calls = np.zeros((NUM_ITERATIONS+1, NUM_IMAGES))
    for iteration in range(NUM_ITERATIONS):
        distances, distances2 = [], []
        for image in range(NUM_IMAGES):
            if 'iterations' not in raw[image]:
                continue
            x_star = raw[image]['original']
            x_t = raw[image]['progression'][iteration]['binary']
            label = raw[image]['true_label']
            distance = np.linalg.norm(x_star - x_t) ** 2 / 784 / 1 ** 2
            probs = model.get_probs([x_t])
            true_prob = probs[0][label]
            abs_error[iteration] += true_prob
            distances.append(distance)
            if i == 2:
                adv_prob = np.max(probs[0][np.arange(10) != label])
                if (true_prob > adv_prob):
                    x_tt = 2*x_t-x_star
                    while True:
                        res = search_boundary(x_t, x_tt,theta_det, label)
                        if np.any(res != x_tt):
                            x_tt = res
                            break
                        x_tt = res
                else:
                    x_tt = search_boundary(x_star, x_t, theta_det, label)
                distance2 = np.linalg.norm(x_star - x_tt) ** 2 / 784 / 1 ** 2
                probs = model.get_probs([x_tt])
                true_prob2 = probs[0][label]
                distances2.append(distance2)
                abs_error2[iteration] += true_prob2

            # Model Calls
            grad_calls[0][image] = raw[image]['model_calls']['initialization']
            bin_calls[0][image] = raw[image]['model_calls']['projection']
            step_calls[0][image] = None
            details = raw[image]['model_calls']['iters']
            grad_calls[iteration+1][image] = details[iteration]['approx_grad']
            step_calls[iteration+1][image] = details[iteration]['step_search']
            bin_calls[iteration+1][image] = details[iteration]['binary']
        medians.append(np.median(distances))
        if i==2:
            medians2.append(np.median(distances2))
    total_grad_calls = grad_calls.mean(axis=1)
    total_step_calls = step_calls.mean(axis=1)
    total_bin_calls = bin_calls.mean(axis=1)

    plot1series1.append(medians)
    plot1series2.append(abs_error/NUM_IMAGES)
    plot2series.append(total_bin_calls)
    if i==2:
        plot1series1.append(medians2)
        plot1series2.append(abs_error2 / NUM_IMAGES)
        plot2series.append(total_bin_calls)

# labels = ['HSJA', 'HSJA (Our Theta)', 'Our Attack', 'Our Attack (with x_t projected to boundary)',
#           'Our Attack (prior disabled)', 'Our Attack (grad step det)']
plt.figure(figsize=(12, 16))
plt.suptitle('Debugging Attack on Deterministic Model with 25 random images')
N = 3
ax1 = plt.subplot(N, 1, 1)
ax1.set_xlabel('Iterations of the Attack')
ax1.set_ylabel('Median L2 Distance')
ax2 = ax1.twinx()
ax2.set_ylabel('Probability of True Label')
for i, s in enumerate(plot1series1):
    ax1.plot(s, label=labels[i])
for i, s in enumerate(plot1series2):
    ax2.plot(s, '--', label=labels[i])
ax1.grid()
ax1.legend(loc='center right')


ax1 = plt.subplot(N, 1, 2)
ax1.set_xlabel('Iterations of the Attack')
ax1.set_ylabel('Average Model Calls')
for i, s in enumerate(plot2series):
    ax1.plot(range(1, NUM_ITERATIONS+2), s, label=labels[i])
ax1.grid()
ax1.legend()

ax1 = plt.subplot(N, 1, 3)
ax1.set_xlabel('Average Model Calls')
ax1.set_ylabel('Median L2 Distance')
for i, s in enumerate(plot2series):
    ax1.plot(s[1:], plot1series1[i], label=labels[i])
ax1.grid()
ax1.legend()

plt.savefig(image_path)
pass