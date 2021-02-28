import matplotlib.pylab as plt
import numpy as np
import torch
from tracker import DiaryPage, Diary
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

plot_distance()
pass
