import matplotlib.pylab as plt
import numpy as np
import torch
from tracker import DiaryPage, Diary
from model_factory import get_models_from_file
from matplotlib import rc
import math

from img_utils import get_device

OUT_DIR = 'thesis'
PLOTS_DIR = f'{OUT_DIR}/plots_multimodel/'
device = get_device()
d = 28*28


def read_dump(path, raw=False, aa=False, aaa=False):
    if raw:
        filepath = f'{OUT_DIR}/{path}/raw_data.pkl'
    elif aa:
        filepath = f'{OUT_DIR}/{path}/crunched_aa.pkl'
    elif aaa:
        filepath = f'{OUT_DIR}/{path}/crunched_aaa.pkl'
    else:
        filepath = f'{OUT_DIR}/{path}/crunched.pkl'
    raw = torch.load(open(filepath, 'rb'), map_location=device)
    return raw


def plot_distance():
    plt.figure(figsize=(10, 7))
    n_models_arr = [1,2,5,10,20,40,75]
    for n_models in n_models_arr:
        exp_name = f'psj_models_{n_models}'
        raw = read_dump(exp_name, raw=True)
        NUM_IMAGES = len(raw)
        NUM_ITERATIONS = len(raw[0].iterations)
        D = torch.zeros(size=(NUM_ITERATIONS+1, NUM_IMAGES))
        for image in range(NUM_IMAGES):
            diary: Diary = raw[image]
            x_star = diary.original
            D[0, image] = torch.norm(diary.initial_projection - x_star) / math.sqrt(d)
            for iteration in range(NUM_ITERATIONS):
                page: DiaryPage = diary.iterations[iteration]
                x_t = page.bin_search
                D[iteration+1, image] = torch.norm(x_t - x_star) / math.sqrt(d)
        plt.plot(np.median(D, axis=1), label=f'models={n_models}')
    plt.grid()
    plt.legend()
    plt.xlabel('iteration')
    plt.ylabel('median $l_2$ distance')
    plt.savefig(f'{PLOTS_DIR}/distance_vs_iteration.pdf', bbox_inches='tight', pad_inches=.02)

    plt.figure(figsize=(10, 7))
    n_models_arr = [1,2,5,10,20,40,75]
    for n_models in n_models_arr:
        exp_name = f'psj_models_{n_models}'
        raw = read_dump(exp_name, raw=True)
        NUM_IMAGES = len(raw)
        NUM_ITERATIONS = len(raw[0].iterations)
        D = torch.zeros(size=(NUM_ITERATIONS+1, NUM_IMAGES))
        C = torch.zeros(size=(NUM_ITERATIONS+1, NUM_IMAGES))
        for image in range(NUM_IMAGES):
            diary: Diary = raw[image]
            x_star = diary.original
            D[0, image] = torch.norm(diary.initial_projection - x_star) / math.sqrt(d)
            C[0, image] = diary.calls_initial_bin_search
            for iteration in range(NUM_ITERATIONS):
                page: DiaryPage = diary.iterations[iteration]
                x_t = page.bin_search
                D[iteration+1, image] = torch.norm(x_t - x_star) / math.sqrt(d)
                C[iteration + 1, image] = page.calls.bin_search
        plt.plot(np.median(C, axis=1), np.median(D, axis=1), label=f'models={n_models}')
    plt.grid()
    plt.legend()
    plt.xlabel('median model calls')
    plt.ylabel('median $l_2$ distance')
    plt.savefig(f'{PLOTS_DIR}/distance_vs_calls.pdf', bbox_inches='tight', pad_inches=.02)


def plot_sigmoids():
    def interpolation(x_star, x_t, alpha):
        return (1 - alpha) * x_star + alpha * x_t

    def sigmoid_estimate(x, s, t, e):
        s, t, e = s.numpy(), t.numpy(), e.numpy()
        sigmoid = lambda z: .5 * np.tanh(2. * z) + .5
        return e + (1 - 2 * e) * sigmoid(s * (x - t))

    n_models = 75
    keys_path = 'data/model_dumps/filtered_models.txt'
    models = get_models_from_file(keys_path, 'mnist', 'deterministic', 0, 1, 'cpu', n_models=n_models)
    exp_name = f'psj_models_{n_models}'
    TARGET_IMAGE = 6
    raw = read_dump(exp_name, raw=True)
    plt.figure(figsize=(10, 30))
    ticks = [-1, 0, 2, 4, 8, 16, 20, 21, 22, 24]
    for i, iteration in enumerate(ticks):
        NUM_IMAGES = len(raw)
        for image in range(NUM_IMAGES):
            if image is not TARGET_IMAGE:
                continue
            diary: Diary = raw[image]
            label = diary.true_label
            if iteration == -1:
                x_tilde = diary.initial_image
                ims = diary.init_infomax
                s, t, e = ims.s, ims.tmap, ims.e
            else:
                page: DiaryPage = diary.iterations[iteration]
                x_tilde = page.opposite
                ims = page.info_max_stats
                s, t, e = ims.s, ims.tmap, ims.e
            x_star = diary.original
            search_space = np.linspace(0, 1, 301)
            points = []
            for alpha in search_space:
                point = interpolation(x_star, x_tilde, 1 - alpha)
                points.append(point)
            points = torch.stack(points)
            num_correct = torch.zeros(n_models, len(points))
            for ii, model in enumerate(models):
                probs_vec = model.get_probs(points)
                is_correct = [torch.argmax(x) == label for x in probs_vec]
                num_correct[ii] = torch.tensor(is_correct)
            prob_correct = torch.sum(num_correct, axis=0) / n_models
            plt.subplot((len(ticks) + 1) / 2, 2, i + 1)
            plt.plot(search_space, prob_correct, label='true')
            estimates = sigmoid_estimate(search_space, s, t, e)
            plt.plot(search_space, estimates, label='estimate')
            plt.ylabel('Probability of True Label')
            plt.xlabel('Search Space')
            y = 0.03 * max(prob_correct) + 0.97 * min(prob_correct)
            plt.text(1.0, y, '$x_\\star$')
            if iteration is -1:
                plt.text(0.0, y, '$x_{init}$')
            else:
                # plt.text(0.0, y, '$\\hat{x}_t$')
                # plt.text(0.33, y, '$\\tilde{x}_t$')
                plt.text(0.0, y, '$\\tilde{x}_t$')
            plt.grid()
            dist = np.abs(x_star.squeeze() - x_tilde.squeeze()).max().float()
            dist_str = '{}'.format(np.round(float(dist), 2))
            plt.legend()
            plt.title("Iteration {}, $||\\tilde{{x}}_t-x_\\star||$={}".format(iteration, dist_str))
    plt.savefig(f'{PLOTS_DIR}/sigmoid.pdf', bbox_inches='tight', pad_inches=.02)

plot_distance()
# plot_sigmoids()
pass
