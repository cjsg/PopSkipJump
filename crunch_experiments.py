import sys
import torch
from tqdm import tqdm
from model_factory import get_model
from img_utils import get_device
from tracker import Diary, DiaryPage

OUT_DIR = 'aistats'
exp_name = sys.argv[1]
flip_prob = float(exp_name.split('_')[-3])
noise = exp_name.split('_')[-5]
beta = float(exp_name.split('_')[-6])
device = get_device()
NUM_ITERATIONS = 32
NUM_IMAGES = 100
eps = list(range(1, 6))
theta_det = 1 / (28 * 28 * 28)


def read_dump(path):
    filepath = f'{OUT_DIR}/{path}/raw_data.pkl'
    raw = torch.load(open(filepath, 'rb'), map_location=device)
    return raw


raw = read_dump(exp_name)
model = get_model(key='mnist_noman', dataset='mnist', beta=beta)
model.model = model.model.to(device)


def search_boundary(x_star, x_t, theta_det, true_label):
    high, low = 1, 0
    while high - low > theta_det:
        mid = (high + low) / 2.0
        x_mid = (1 - mid) * x_star + mid * x_t
        pred = torch.argmax(model.get_probs(x_mid[None])[0])
        if pred == true_label:
            low = mid
        else:
            high = mid
    out = (1 - high) * x_star + high * x_t
    return out


def project(x_star, x_t, label, theta_det):
    probs = model.get_probs(x_t[None])
    if torch.argmax(probs[0]) == label:
        c = 0.25
        while True:
            x_tt = x_t + c * (x_t - x_star) / torch.norm(x_t - x_star)
            pred = torch.argmax(model.get_probs(x_tt[None])[0])
            if pred != label:
                x_tt = search_boundary(x_t, x_tt, theta_det, label)
                break
            c += c
    else:
        x_tt = search_boundary(x_star, x_t, theta_det, label)
    return x_tt


D = torch.zeros(size=(NUM_ITERATIONS + 1, NUM_IMAGES), device=device)
D_OUT = torch.zeros_like(D, device=device)
MC = torch.zeros_like(D, device=device)
AA = torch.zeros(size=(len(eps), NUM_ITERATIONS + 1, NUM_IMAGES), device=device)
P = torch.zeros_like(D, device=device)
P_OUT = torch.zeros_like(D, device=device)

for iteration in tqdm(range(NUM_ITERATIONS)):
    for image in range(NUM_IMAGES):
        diary: Diary = raw[image]
        x_star = diary.original
        label = diary.true_label
        if iteration == 0:
            x_0 = diary.initial_projection
            x_00 = project(x_star, x_0, label, theta_det)
            p_0 = torch.argmax(model.get_probs(x_0[None])[0]) == label
            p_00 = torch.argmax(model.get_probs(x_00[None])[0]) == label
            D[0, image] = torch.norm(x_star - x_00) ** 2 / 784 / 1 ** 2
            D_OUT[0, image] = -1
            MC[0, image] = diary.calls_initial_bin_search
            P[0, image] = p_00
            P_OUT[0, image] = p_0

        page: DiaryPage = diary.iterations[iteration]
        calls = page.calls.bin_search
        x_t = page.bin_search
        x_tt = project(x_star, x_t, label, theta_det)
        p_t = torch.argmax(model.get_probs(x_t[None])[0]) == label

        D[iteration + 1, image] = torch.norm(x_star - x_tt) ** 2 / 784 / 1 ** 2
        if exp_name.startswith('psj'):
            D_OUT[iteration + 1, image] = D[iteration, image]
        else:
            D_OUT[iteration + 1, image] = page.distance
        MC[iteration + 1, image] = calls

        s_, e_ = page.info_max_stats.s, page.info_max_stats.e
        t_ = (page.bin_search[0, 0] - x_star[0, 0]) / (page.opposite[0, 0] - x_star[0, 0])
        y_ = (0.3 - e_) / (1 - 2 * e_)
        x_ = torch.log(y_ / (1 - y_)) / (4 * s_) + t_
        x_est = (1-t_) * x_star + t_ * page.opposite
        P[iteration + 1, image] = p_t
        P_OUT[iteration + 1, image] = torch.argmax(model.get_probs(x_est[None])[0]) == label


        for j in range(len(eps)):
            x_adv = x_star + eps[j] * (x_tt - x_star) / torch.norm(x_tt - x_star)
            p_adv = model.get_probs(x_adv[None])[0]
            if noise == 'bayesian':
                AA[j, iteration + 1, image] = p_adv[label]
            elif noise == 'deterministic':
                AA[j, iteration + 1, image] = (torch.argmax(p_adv) == label) * 1.0
            else:
                p_temp = torch.ones_like(p_adv) * flip_prob / (p_adv.shape[0] - 1)
                pred = torch.argmax(p_adv)
                p_temp[pred] = 1 - flip_prob
                AA[j, iteration + 1, image] = p_temp[label]

dump = {
    'border_distance': D,
    'attack_out_distance': D_OUT,
    'model_calls': MC,
    'adv_acc': AA,
    'prob_true_label': P,
    'prob_true_label_out': P_OUT
}
torch.save(dump, open(f'{OUT_DIR}/{exp_name}/crunched.pkl', 'wb'))
