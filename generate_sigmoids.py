import pickle
import numpy as np
import matplotlib.pylab as plt
from model_factory import get_model
from scipy.optimize import minimize

TITLE = 'Infomax Progression'
NUM_ITERATIONS = 32
NUM_IMAGES = 50
NOISE = 'bayesian'
exp_name = 'hsja_on_det_model'
image_path = 'adv/del_later.pdf'


model = get_model(key='mnist_noman', dataset='mnist')


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

    p0 = np.ones(4)
    bnds = ((0.0, 1.0), (0.0, 1.0), (0.0, 100), (-10, 10))
    solution = minimize(objective, p0, method='SLSQP', bounds=bnds)
    p = solution.x
    return p

    # y_hat = [pred(p, i) for i in range(len(x))]
    # plt.figure(figsize=(7, 7))
    # plt.plot(x, y_star)
    # plt.plot(x, y_hat)
    # plt.grid()
    # plt.savefig(image_path)


raws = read_dump(exp_name)
data = np.zeros(shape=(NUM_ITERATIONS+1, NUM_IMAGES, 4))
for i, raw in enumerate(raws):
    for iteration in range(NUM_ITERATIONS+1):
        print ('iteration {}'.format(iteration))
        for image in range(NUM_IMAGES):
            if 'iterations' not in raw[image]:
                continue
            label = raw[image]['true_label']
            original = raw[image]['original']
            details = raw[image]['progression'][iteration]
            app_image = details['approx_grad']
            alphas = np.linspace(0, 1, 1000)
            projections = []
            for alpha in alphas:
                projections.append(original * alpha + app_image * (1-alpha))
            opp_probs = model.get_probs(projections)[:, label]
            res = solve_opt(alphas, opp_probs)
            data[iteration, image] = res
np.save("saved.npy", data)
pass