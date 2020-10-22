import matplotlib.pylab as plt
import numpy as np
import sys
import torch
from tracker import DiaryPage, Diary
from matplotlib import rc

from img_utils import get_device

OUT_DIR = 'aistats'
device = get_device()


def read_dump(path):
    filepath = f'{OUT_DIR}/{path}/raw_data.pkl'
    raw = torch.load(open(filepath, 'rb'), map_location=device)
    return raw


def plot_distance(exp_name):
    plt.figure(figsize=(10, 7))
    raw = read_dump(exp_name)
    NUM_IMAGES = len(raw)
    NUM_ITERATIONS = len(raw[0].iterations)
    D = torch.zeros(size=(NUM_ITERATIONS+1, NUM_IMAGES))
    for image in range(NUM_IMAGES):
        diary: Diary = raw[image]
        x_star = diary.original
        D[0, image] = torch.norm(diary.initial_projection - x_star)
        for iteration in range(NUM_ITERATIONS):
            page: DiaryPage = diary.iterations[iteration]
            x_t = page.bin_search
            D[iteration+1, image] = torch.norm(x_t - x_star)
    plt.plot(np.median(D, axis=1))
    plt.grid()
    plt.xlabel('iteration')
    plt.ylabel('median $l_2$ distance')
    plt.savefig(f'{OUT_DIR}/plot.pdf', bbox_inches='tight', pad_inches=.02)
    print(f"Saved plot at: {OUT_DIR}/plot.pdf")


rc('text', usetex=True)
rc('text.latex', preamble=[r'\usepackage{amsfonts}'])
rc('font', size=14)  # default: 10 -> choose size depending on figsize
rc('font', family='STIXGeneral')
rc('legend', fontsize=16)
plt.tight_layout(h_pad=0, w_pad=.5)
plot_distance(sys.argv[1])