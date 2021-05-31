import matplotlib.pylab as plt
import numpy as np
import torch
from tracker import DiaryPage, Diary
from model_factory import get_models_from_file
from matplotlib import rc
import math
from tqdm import tqdm
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


def interpolation(x_star, x_t, alpha):
    return (1 - alpha) * x_star + alpha * x_t


def prediction(models, x, true_label):
    freq = {}
    for model in models:
        pred = torch.argmax(model.get_probs(x[None])[0]).item()
        if pred not in freq:
            freq[pred] = 0
        freq[pred] += 1
    # if freq[true_label]
    max_f, max_l = -1, None
    for l in freq:
        if freq[l] > max_f:
            max_f, max_l = freq[l], l
    return max_l


def search_boundary(x_star, x_t, theta_det, true_label, models):
    high, low = 1, 0
    while high - low > theta_det:
        mid = (high + low) / 2.0
        x_mid = interpolation(x_star, x_t, mid)
        pred = prediction(models, x_mid, true_label)
        if pred == true_label:
            low = mid
        else:
            high = mid
    out = interpolation(x_star, x_t, high)
    return out


def project(x_star, x_t, label, theta_det, models):
    pred = prediction(models, x_t, label)
    if pred == label:
        c = 0.25
        while True:
            x_tt = x_t + c * (x_t - x_star) / torch.norm(x_t - x_star)
            x_tt = torch.clamp(x_tt, 0, 1)
            if torch.all(torch.logical_or(x_tt == 1, x_tt == 0)).item() or c > 2**20:
                break
            pred = prediction(models, x_tt, label)
            if pred != label:
                x_tt = search_boundary(x_t, x_tt, theta_det, label, models)
                break
            c += c
    else:
        x_tt = search_boundary(x_star, x_t, theta_det, label, models)
    return x_tt


def plot_distance():
    rc('text', usetex=True)
    rc('text.latex', preamble=[r'\usepackage{amsfonts}'])
    rc('font', size=20)  # default: 10 -> choose size depending on figsize
    rc('font', family='STIXGeneral')
    rc('legend', fontsize=20)
    plt.tight_layout(h_pad=0, w_pad=.5)
    plt.figure(figsize=(10, 7))
    n_models_arr = [1,2,5,10,20,40,75]
    for n_models in n_models_arr:
        models = get_models_from_file('data/model_dumps/filtered_models.txt', 'mnist', 'deterministic',
                                      device=get_device(), n_models=n_models)
        exp_name = f'psj_models_{n_models}'
        raw = read_dump(exp_name, raw=True)
        NUM_IMAGES = len(raw)
        NUM_ITERATIONS = len(raw[0].iterations)
        D = torch.zeros(size=(NUM_ITERATIONS+1, NUM_IMAGES))
        for image in tqdm(range(NUM_IMAGES)):
            diary: Diary = raw[image]
            x_star = diary.original
            label = diary.true_label
            D[0, image] = torch.norm(diary.initial_projection - x_star) / math.sqrt(d)
            for iteration in range(NUM_ITERATIONS):
                page: DiaryPage = diary.iterations[iteration]
                x_t = page.bin_search
                x_tt = project(x_star, x_t, label, 1.0 / 28*28*28, models)
                D[iteration+1, image] = torch.norm(x_tt - x_star) / math.sqrt(d)
        plt.plot(np.median(D, axis=1), label=f'models={n_models}')
    plt.grid()
    plt.legend()
    plt.xlabel('iterations')
    plt.ylabel('median argmax border distance')
    plt.savefig(f'{PLOTS_DIR}/distance_vs_iteration.pdf', bbox_inches='tight', pad_inches=.02)

    # plt.figure(figsize=(10, 7))
    # n_models_arr = [1,2,5,10,20,40,75]
    # for n_models in n_models_arr:
    #     exp_name = f'psj_models_{n_models}'
    #     raw = read_dump(exp_name, raw=True)
    #     NUM_IMAGES = len(raw)
    #     NUM_ITERATIONS = len(raw[0].iterations)
    #     D = torch.zeros(size=(NUM_ITERATIONS+1, NUM_IMAGES))
    #     C = torch.zeros(size=(NUM_ITERATIONS+1, NUM_IMAGES))
    #     for image in range(NUM_IMAGES):
    #         diary: Diary = raw[image]
    #         x_star = diary.original
    #         D[0, image] = torch.norm(diary.initial_projection - x_star) / math.sqrt(d)
    #         C[0, image] = diary.calls_initial_bin_search
    #         for iteration in range(NUM_ITERATIONS):
    #             page: DiaryPage = diary.iterations[iteration]
    #             x_t = page.bin_search
    #             D[iteration+1, image] = torch.norm(x_t - x_star) / math.sqrt(d)
    #             C[iteration + 1, image] = page.calls.bin_search
    #     plt.plot(np.median(C, axis=1), np.median(D, axis=1), label=f'models={n_models}')
    # plt.grid()
    # plt.legend()
    # plt.xlabel('median model calls')
    # plt.ylabel('median $l_2$ distance')
    # plt.savefig(f'{PLOTS_DIR}/distance_vs_calls.pdf', bbox_inches='tight', pad_inches=.02)

def plot_distance_arch():
    rc('text', usetex=True)
    rc('text.latex', preamble=[r'\usepackage{amsfonts}'])
    rc('font', size=20)  # default: 10 -> choose size depending on figsize
    rc('font', family='STIXGeneral')
    rc('legend', fontsize=20)
    plt.figure(figsize=(10, 7))
    n_models_arr = [1,2,5,10]
    for n_models in n_models_arr:
        models = get_models_from_file(f'training/data/model_dumps/filtered_models_{n_models}each.txt', 'mnist', 'deterministic',
                                      device=get_device(), n_models=n_models*4)
        exp_name = f'psj_models_{n_models}_each'
        raw = read_dump(exp_name, raw=True)
        NUM_IMAGES = len(raw)
        NUM_ITERATIONS = len(raw[0].iterations)
        D = torch.zeros(size=(NUM_ITERATIONS+1, NUM_IMAGES))
        for image in tqdm(range(NUM_IMAGES)):
            diary: Diary = raw[image]
            x_star = diary.original
            label = diary.true_label
            D[0, image] = torch.norm(diary.initial_projection - x_star) / math.sqrt(d)
            for iteration in range(NUM_ITERATIONS):
                page: DiaryPage = diary.iterations[iteration]
                x_t = page.bin_search
                x_tt = project(x_star, x_t, label, 1.0 / 28*28*28, models)
                D[iteration+1, image] = torch.norm(x_tt - x_star) / math.sqrt(d)
        plt.plot(np.median(D, axis=1), label=f'{n_models} models/architecture')
    plt.grid()
    plt.legend()
    plt.xlabel('iterations')
    plt.ylabel('median argmax border distance')
    plt.savefig(f'{PLOTS_DIR}/distance_vs_iteration_arch.pdf', bbox_inches='tight', pad_inches=.02)

    # plt.figure(figsize=(10, 7))
    # n_models_arr = [1,2,5,10]
    # for n_models in n_models_arr:
    #     exp_name = f'psj_models_{n_models}_each'
    #     raw = read_dump(exp_name, raw=True)
    #     NUM_IMAGES = len(raw)
    #     NUM_ITERATIONS = len(raw[0].iterations)
    #     D = torch.zeros(size=(NUM_ITERATIONS+1, NUM_IMAGES))
    #     C = torch.zeros(size=(NUM_ITERATIONS+1, NUM_IMAGES))
    #     for image in range(NUM_IMAGES):
    #         diary: Diary = raw[image]
    #         x_star = diary.original
    #         D[0, image] = torch.norm(diary.initial_projection - x_star) / math.sqrt(d)
    #         C[0, image] = diary.calls_initial_bin_search
    #         for iteration in range(NUM_ITERATIONS):
    #             page: DiaryPage = diary.iterations[iteration]
    #             x_t = page.bin_search
    #             D[iteration+1, image] = torch.norm(x_t - x_star) / math.sqrt(d)
    #             C[iteration + 1, image] = page.calls.bin_search
    #     plt.plot(np.median(C, axis=1), np.median(D, axis=1), label=f'{n_models} models/architecture')
    # plt.grid()
    # plt.legend()
    # plt.xlabel('median model calls')
    # plt.ylabel('median $l_2$ distance')
    # plt.savefig(f'{PLOTS_DIR}/distance_vs_calls_arch.pdf', bbox_inches='tight', pad_inches=.02)


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
    plt.figure(figsize=(10, 16))
    # ticks = [-1, 0, 2, 4, 8, 16, 20, 21, 22, 24]
    ticks = [-1, 0, 4, 8, 16, 24]
    for i, iteration in enumerate(ticks):
        NUM_IMAGES = len(raw)
        for image in range(NUM_IMAGES):
            if image is not TARGET_IMAGE:
                continue
            diary: Diary = raw[image]
            label = diary.true_label
            if iteration == -1:
                x_hat = diary.initial_image
                ims = diary.init_infomax
                s, t, e = ims.s, ims.tmap, ims.e
            else:
                page: DiaryPage = diary.iterations[iteration]
                x_hat = page.opposite
                ims = page.info_max_stats
                s, t, e = ims.s, ims.tmap, ims.e
            x_star = diary.original
            search_space = np.linspace(0, 1, 301)
            points = []
            for alpha in search_space:
                point = interpolation(x_star, x_hat, 1 - alpha)
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
                plt.text(0.0, y, '$\\hat{x}_t$')
                plt.text(0.33, y, '$\\tilde{x}_t$')
                # plt.text(0.0, y, '$\\tilde{x}_t$')
            plt.grid()
            dist = np.linalg.norm(x_star.squeeze() - x_hat.squeeze())
            dist_str = '{}'.format(np.round(float(dist), 2))
            plt.legend()
            plt.title("Iteration {}, $||\\hat{{x}}_t-x_\\star||$={}".format(iteration, dist_str))
    plt.savefig(f'{PLOTS_DIR}/sigmoid.pdf', bbox_inches='tight', pad_inches=.02)


def plot_sigmoids_arch():
    def interpolation(x_star, x_t, alpha):
        return (1 - alpha) * x_star + alpha * x_t

    def sigmoid_estimate(x, s, t, e):
        s, t, e = s.numpy(), t.numpy(), e.numpy()
        sigmoid = lambda z: .5 * np.tanh(2. * z) + .5
        return e + (1 - 2 * e) * sigmoid(s * (x - t))

    n_models = 40
    keys_path = 'training/data/model_dumps/filtered_models_10each.txt'
    models = get_models_from_file(keys_path, 'mnist', 'deterministic', 0, 1, 'cpu', n_models=n_models)
    exp_name = f'psj_models_{n_models}'
    TARGET_IMAGE = 6
    raw = read_dump(exp_name, raw=True)
    plt.figure(figsize=(10, 16))
    # ticks = [-1, 0, 2, 4, 8, 16, 20, 21, 22, 24]
    ticks = [-1, 0, 4, 8, 16, 24]
    for i, iteration in enumerate(ticks):
        NUM_IMAGES = len(raw)
        for image in range(NUM_IMAGES):
            if image is not TARGET_IMAGE:
                continue
            diary: Diary = raw[image]
            label = diary.true_label
            if iteration == -1:
                x_hat = diary.initial_image
                ims = diary.init_infomax
                s, t, e = ims.s, ims.tmap, ims.e
            else:
                page: DiaryPage = diary.iterations[iteration]
                x_hat = page.opposite
                ims = page.info_max_stats
                s, t, e = ims.s, ims.tmap, ims.e
            x_star = diary.original
            search_space = np.linspace(0, 1, 301)
            points = []
            for alpha in search_space:
                point = interpolation(x_star, x_hat, 1 - alpha)
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
                plt.text(0.0, y, '$\\hat{x}_t$')
                plt.text(0.33, y, '$\\tilde{x}_t$')
                # plt.text(0.0, y, '$\\tilde{x}_t$')
            plt.grid()
            dist = np.linalg.norm(x_star.squeeze() - x_hat.squeeze())
            dist_str = '{}'.format(np.round(float(dist), 2))
            plt.legend()
            plt.title("Iteration {}, $||\\hat{{x}}_t-x_\\star||$={}".format(iteration, dist_str))
    plt.savefig(f'{PLOTS_DIR}/sigmoid-arch.pdf', bbox_inches='tight', pad_inches=.02)


# plot_distance()
plot_distance_arch()
# plot_sigmoids()
# plot_sigmoids_arch()
pass
