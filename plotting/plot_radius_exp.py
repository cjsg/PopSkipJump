import pickle
import numpy as np
import matplotlib.pylab as plt
from model_factory import get_model

NUM_ITERATIONS = 30
NUM_IMAGES = 1000

FLIP_PROB = 0.2

model = get_model(key='mnist_noman', dataset='mnist')

G = np.round(10 ** np.linspace(0, -2, num=9), 2)
G = [G[i] for i in [0,1,2,4,8]]
labels = ['$\\beta$=%.2f' % g for g in G]
G = list(G) + [0.00127]
labels.append('$\\beta$=$d^{-1}$=1.27e-3')

def read_dumps():
    raws = []
    for gamma in G:
        folder = 'det_30_1000_g{}'.format(gamma)
        filepath = 'adv/{}/raw_data.pkl'.format(folder)
        raws.append(pickle.load(open(filepath, 'rb')))
    return raws


raws = read_dumps()
import matplotlib.pylab as plt
from matplotlib import rc
rc('text', usetex=True)
rc('text.latex', preamble=[r'\usepackage{amsfonts}'])
rc('font', size=14)  # default: 10 -> choose size depending on figsize
rc('font', family='STIXGeneral')
rc('legend', fontsize=14)
plt.tight_layout(h_pad=0, w_pad=.5)

fig, ax1 = plt.subplots(figsize=(8, 5))
ax1.set_xlabel('iterations')
ax1.set_ylabel('median border distance')
ax1.set_yscale('log')
# ax2 = ax1.twinx()
# ax2.set_ylabel('Proportion of Non-adversarial samples')


for i, raw in enumerate(raws):
    # adv_count = np.zeros(NUM_ITERATIONS, )
    medians = []
    for iteration in range(NUM_ITERATIONS):
        distances = []
        for image in range(NUM_IMAGES):
            if 'iterations' not in raw[image]:
                continue
            distance = raw[image]['iterations'][iteration]['distance']
            # adversarial = raw[image]['iterations'][iteration]['perturbed']
            # probs = model.get_probs([adversarial])
            label = raw[image]['true_label']
            # adv_prob = np.max(probs[0][np.arange(10) != label])
            # if np.argmax(probs[0]) != label:
            #     adv_count[iteration] += 1
            distances.append(distance)
        medians.append(np.median(distances))
    ax1.plot(range(1, 31), medians, label=labels[i], linewidth=0.9)
    # ax2.plot(1 - adv_count / NUM_IMAGES, '--', label='g = {}'.format(G[i]))

ax1.grid()
# plt.title('Radius Size Experiment with Deterministic Model (1000 images)'
#           '\ntheta = g / sqrt(d)'
#           '\ndelta = theta * sqrt(d) * L2_dist(t)')
ax1.legend()
plt.savefig('adv/radius_exp_1000.pdf', bbox_inches='tight', pad_inches=.02)
pass
