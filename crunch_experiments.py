import sys
import torch
from tqdm import tqdm
from model_factory import get_model
from img_utils import get_device
import math
from tracker import Diary, DiaryPage

OUT_DIR = 'thesis'
exp_name = sys.argv[1]
dataset = sys.argv[2]
flip_prob = float(exp_name.split('_')[-3])
noise = exp_name.split('_')[-5]
beta = float(exp_name.split('_')[-6])
distance_metric = str(exp_name.split('_')[-8])
device = get_device()
NUM_ITERATIONS = 32
NUM_IMAGES = int(exp_name.split('_')[-1])
eps = list(range(1, 6))
if dataset == 'cifar10':
    d = 32*32*3
    model = get_model(key='cifar10', dataset=dataset, beta=beta)
    model_noisy = get_model(key='cifar10', dataset=dataset, noise=noise,
                            smoothing_noise=0.01, crop_size=26, drop_rate=0.5)
else:
    d = 28*28
    model = get_model(key='mnist_noman', dataset=dataset, beta=beta)
    model_noisy = get_model(key='mnist_noman', dataset=dataset, noise=noise,
                            smoothing_noise=0.01, crop_size=26, drop_rate=0.5)
if distance_metric == 'l2':
    theta_det = 1 / (d * math.sqrt(d))
elif distance_metric == 'linf':
    theta_det = 1 / (d * d)


def read_dump(path):
    filepath = f'{OUT_DIR}/{path}/raw_data.pkl'
    raw = torch.load(open(filepath, 'rb'), map_location=device)
    return raw


raw = read_dump(exp_name)
model.model = model.model.to(device)
model_noisy.model = model_noisy.model.to(device)


def interpolation(x_star, x_t, alpha):
    if distance_metric == 'l2':
        x_mid = (1 - alpha) * x_star + alpha * x_t
    elif distance_metric == 'linf':
        dist_linf = torch.max(torch.abs(x_star - x_t))
        min_limit = x_star - alpha * dist_linf
        max_limit = x_star + alpha * dist_linf
        x_mid = torch.where(x_t > max_limit, max_limit, x_t)
        x_mid = torch.where(x_mid < min_limit, min_limit, x_mid)
    return x_mid


def smoothing_output(x, true_label, samples=50):
    dim = [samples] + [1] * x.dim()
    x = x.unsqueeze(dim=0).repeat(*(dim))
    pred = model_noisy.ask_model(x)
    correct_pred = torch.sum(pred == true_label).float()
    p = correct_pred / samples
    if p >= 0.5:
        return true_label
    else:
        return (true_label + 1) % 10


def search_boundary(x_star, x_t, theta_det, true_label, smoothing=False):
    high, low = 1, 0
    while high - low > theta_det:
        mid = (high + low) / 2.0
        x_mid = interpolation(x_star, x_t, mid)
        if smoothing:
            pred = smoothing_output(x_mid, true_label)
        else:
            pred = torch.argmax(model.get_probs(x_mid[None])[0])
        if pred == true_label:
            low = mid
        else:
            high = mid
    out = interpolation(x_star, x_t, high)
    return out


def compute_distance(x1, x2):
    if distance_metric == "l2":
        return torch.norm(x1 - x2) / math.sqrt(d)
    elif distance_metric == "linf":
        return torch.max(torch.abs(x1 - x2))


def project(x_star, x_t, label, theta_det, smoothing=False):
    if smoothing:
        pred = smoothing_output(x_t, label)
    else:
        pred = torch.argmax(model.get_probs(x_t[None])[0])
    if pred == label:
        c = 0.25
        while True:
            x_tt = x_t + c * (x_t - x_star) / torch.norm(x_t - x_star)
            x_tt = torch.clamp(x_tt, 0, 1)
            if torch.all(torch.logical_or(x_tt == 1, x_tt == 0)).item() or c > 2**20:
                break
            if smoothing:
                pred = smoothing_output(x_tt, label)
            else:
                pred = torch.argmax(model.get_probs(x_tt[None])[0])
            if pred != label:
                x_tt = search_boundary(x_t, x_tt, theta_det, label, smoothing)
                break
            c += c
    else:
        x_tt = search_boundary(x_star, x_t, theta_det, label, smoothing)
    return x_tt


D = torch.zeros(size=(NUM_ITERATIONS + 1, NUM_IMAGES), device=device)
D_SMOOTH = torch.zeros_like(D, device=device)
D_VANILLA = torch.zeros_like(D, device=device)
D_OUT = torch.zeros_like(D, device=device)
D_G = torch.zeros_like(D, device=device)
MC = torch.zeros_like(D, device=device)
AA = torch.zeros(size=(len(eps), NUM_ITERATIONS + 1, NUM_IMAGES), device=device)

for iteration in tqdm(range(NUM_ITERATIONS)):
    for image in range(NUM_IMAGES):
        diary: Diary = raw[image]
        x_star = diary.original
        label = diary.true_label
        if iteration == 0:
            x_0 = diary.initial_projection
            x_00 = project(x_star, x_0, label, theta_det)
            x_00_smooth = project(x_star, x_0, label, theta_det, smoothing=True)
            D[0, image] = compute_distance(x_star, x_00)
            D_SMOOTH[0, image] = compute_distance(x_star, x_00_smooth)
            D_VANILLA[0, image] = compute_distance(x_star, x_0)
            D_OUT[0, image] = -1
            MC[0, image] = diary.calls_initial_bin_search

        page: DiaryPage = diary.iterations[iteration]
        calls = page.calls.bin_search
        x_t = page.bin_search
        x_tt = project(x_star, x_t, label, theta_det)
        x_tt_smooth = project(x_star, x_t, label, theta_det, smoothing=True)

        D[iteration + 1, image] = compute_distance(x_star, x_tt)
        D_SMOOTH[iteration + 1, image] = compute_distance(x_star, x_tt_smooth)
        D_VANILLA[iteration + 1, image] = compute_distance(x_star, x_t)
        D_G[iteration+1, image] = compute_distance(x_star, page.approx_grad)
        if exp_name.startswith('psj'):
            D_OUT[iteration + 1, image] = D[iteration, image]
        else:
            D_OUT[iteration + 1, image] = page.distance
        MC[iteration + 1, image] = calls

        for j in range(len(eps)):
            x_adv = x_star + eps[j] * (x_tt - x_star) / torch.norm(x_tt - x_star)
            p_adv = model.get_probs(x_adv[None])[0]
            if noise == 'bayesian':
                AA[j, iteration + 1, image] = p_adv[label]
            elif noise in ['deterministic', 'smoothing']:
                AA[j, iteration + 1, image] = (torch.argmax(p_adv) == label) * 1.0
            else:
                p_temp = torch.ones_like(p_adv) * flip_prob / (p_adv.shape[0] - 1)
                pred = torch.argmax(p_adv)
                p_temp[pred] = 1 - flip_prob
                AA[j, iteration + 1, image] = p_temp[label]

dump = {
    'border_distance': D,
    'border_distance_smooth': D_SMOOTH,
    'vanilla_distance': D_VANILLA,
    'distance_approxgrad': D_G,
    'attack_out_distance': D_OUT,
    'model_calls': MC,
    'adv_acc': AA,
}
torch.save(dump, open(f'{OUT_DIR}/{exp_name}/crunched.pkl', 'wb'))
