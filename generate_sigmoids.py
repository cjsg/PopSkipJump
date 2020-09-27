import pickle
import numpy as np
import sys
import matplotlib.pylab as plt
from model_factory import get_model
from scipy.optimize import minimize

TITLE = 'Infomax Progression'
NUM_ITERATIONS = 32
NUM_IMAGES = 50
BETA = float(sys.argv[3])
FROM = int(sys.argv[1])
TO = int(sys.argv[2])
# TARGET_IMAGE = 10
NOISE = 'bayesian'
exp_name = 'hsja_500_dusra'
image_path = 'adv/del_later_dusra.pdf'


model = get_model(key='mnist_noman', dataset='mnist', beta=BETA)


def read_dump(path):
    raws = []
    filepath = 'adv/{}/raw_data.pkl'.format(path)
    raws.append(pickle.load(open(filepath, 'rb')))
    return raws


def solve_opt(x, y_star):
    def pred(p, i):
        return p[0] + (1 - p[0] - p[1]) / (1 + np.exp(-p[2] * (x[i] - p[3])))

    def objective(p):
        loss = 0.0
        for i in range(len(x)):
            y_hat = pred(p, i)
            loss += (y_star[i] - y_hat)**2
        return loss/len(x)

    def constraint1a(p):
        return pred(p, 0) - y_star[0]

    def constraint1b(p):
        n = len(x)-1
        return pred(p, n) - y_star[n]

    def constraint2(p):
        return 1 - p[0] - p[1]

    p0 = np.array([0, 0, 1, 0.5])
    bnds = ((0.0, 1.0), (0.0, 1.0), (0.0, 100), (-10, 10))
    cons = [{"fun": constraint1a, "type": "eq"},
            {"fun": constraint1b, "type": "eq"},
            {"fun": constraint2, "type": "ineq"}]
    solution = minimize(objective, p0, method='SLSQP', bounds=bnds, constraints=cons)
    p = solution.x

    y_hat = [pred(p, i) for i in range(len(x))]
    # plt.figure(figsize=(7, 7))
    # plt.plot(x, y_star)
    # plt.plot(x, y_hat)
    # plt.grid()
    # plt.savefig(image_path)

    return p, y_hat


raws = read_dump(exp_name)
data = np.zeros(shape=(NUM_ITERATIONS+1, TO-FROM+1, 4))
# plt.figure(figsize=(12, 16))
# plt.suptitle(f'Image {TARGET_IMAGE}')
for i, raw in enumerate(raws):
    for iteration in range(NUM_ITERATIONS+1):
        print ('iteration {}'.format(iteration))
        for image in range(FROM, TO+1):
            if 'iterations' not in raw[image]:
                continue
            label = raw[image]['true_label']
            original = raw[image]['original']
            details = raw[image]['progression'][iteration]
            app_image = details['approx_grad']
            alphas = np.linspace(-1, 1, 2001)
            projections = []
            for alpha in alphas:
                projections.append(original * alpha + app_image * (1-alpha))
            opp_probs = model.get_probs(projections)[:, label]
            # opp_probs = opp_probs/2 + 0.25
            res, y_hat = solve_opt(alphas, opp_probs)
            data[iteration, image-FROM] = res
            # ax1 = plt.subplot(NUM_ITERATIONS/8 + 1, 2, iteration/4 + 1)
            # ax1.plot(alphas, opp_probs)
            # ax1.plot(alphas, y_hat)
            # ax1.set_title(f"Iteration {iteration}")
            # ax1.grid()
# plt.savefig(f"sigmoids_image{TARGET_IMAGE}.pdf")
np.save(f"adv/sigmoids_v2_{BETA}_{FROM}_{TO}.npy", data)
pass