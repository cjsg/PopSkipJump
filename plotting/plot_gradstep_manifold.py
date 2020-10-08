import pickle
import numpy as np
import matplotlib.pylab as plt
from model_factory import get_model

NUM_ITERATIONS = 32
NUM_IMAGES = 10
TARGET_IMAGE = 7
NOISE = 'bayesian'
exp_name = 'eval_exp_wo_step_10'
image_path = 'adv/eval_exp_wo_step_10_manifold_gradstepv2.pdf'


model = get_model(key='mnist_noman', dataset='mnist')


def read_dump(path):
    raws = []
    filepath = 'adv/{}/raw_data.pkl'.format(path)
    raws.append(pickle.load(open(filepath, 'rb')))
    return raws


raws = read_dump(exp_name)

for raw in raws:
    plt.figure(figsize=(10, 15))
    plt.suptitle('Manifold Visualization Grad StepSize\n'
                 'x1=\\tilde{x}_{t+1}\nx2=x_t')
    ticks = [0, 5, 10, 15, 20, 30]
    for i, iteration in enumerate(ticks):
        for image in range(NUM_IMAGES):
            if image is not TARGET_IMAGE:
                continue
            label = raw[image]['true_label']
            progression = raw[image]['progression']
            bin_image = progression[iteration]['binary']
            app_image = progression[iteration+1]['approx_grad']
            dist = np.linalg.norm(bin_image.squeeze() - app_image.squeeze())
            search_space = np.linspace(-1, 2, 301)
            points = []
            for alpha in search_space:
                point = (1 - alpha) * app_image + alpha * bin_image
                points.append(point)
            probs_vec = model.get_probs(points)
            probs = [x[label] for x in probs_vec]
            plt.subplot(len(ticks)/2, 2, i + 1)
            plt.plot(search_space, probs)
            plt.ylabel('Probability of True Label')
            plt.xlabel('Search Space')
            y = 0.03*max(probs) + 0.97*min(probs)
            plt.text(0.0, y, 'x1')
            plt.text(1.0, y, 'x2')
            # plt.text(-1, 0.03, '2x - x*')
            plt.grid()
            dist_str = '{}'.format(np.round((dist**2)/784, 5))
            plt.title("Iteration {}, ||x1-x2||={}".format(iteration, dist_str))
    plt.savefig(image_path)

pass