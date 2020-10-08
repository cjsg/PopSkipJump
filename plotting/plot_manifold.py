import pickle
import numpy as np
import matplotlib.pylab as plt
from model_factory import get_model

NUM_ITERATIONS = 24
NUM_IMAGES = 4
model = get_model(key='mnist_noman', dataset='mnist')


def read_dump():
    folder = 'exp_{}_gsf{}_sf{}_input'.format(NUM_ITERATIONS, 16, 256)
    filepath = 'adv/{}/raw_data.pkl'.format(folder)
    return pickle.load(open(filepath, 'rb'))


raw = read_dump()
for image in range(NUM_IMAGES):
    manifolds = raw[image]['manifold']
    TRUE_LABEL = raw[image]['true_label']
    print(TRUE_LABEL)
    plt.figure(figsize=(10, 12))
    plt.suptitle('Manifold Visualization for Image {}\nSlack=0.10, Freq in BS=256, C=16'.format(image))
    chosen_manfs = [raw[image]['manifold_init']] + [manifolds[k] for k in np.linspace(0, 20, 3).astype(int)]
    for i, manifold in enumerate(chosen_manfs):
        plt.subplot(2, 2, i + 1)
        true_series = []
        adv_series = []
        xaxis = list(sorted(manifold.keys()))
        for x in xaxis:
            probs = manifold[x]
            true_series.append(probs[TRUE_LABEL])
            adv_series.append(np.max(np.delete(probs, TRUE_LABEL)))
        xticks = [2*x-1 for x in xaxis]
        plt.plot(xticks, true_series, label="True Label")
        plt.plot(xticks, adv_series, label="Max of Incorrect Labels")
        plt.ylabel('Probabilities')
        plt.xlabel('Search Space')
        plt.text(0.0, 0.03, 'x')
        plt.text(1.0, 0.03, 'x*')
        plt.text(-1, 0.03, '2x - x*')
        plt.grid()
        plt.legend()
        if i == 0:
            title = 'Initial Binary Search'
        else:
            title = 'End of Iteration {}'.format((i-1)*10)
        plt.title(title)
    plt.savefig('adv/manifold_s0.1_input_{}.pdf'.format(image))
pass
