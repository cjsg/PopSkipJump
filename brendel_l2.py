#!/usr/bin/env python3
"""
A simple example that demonstrates how to run a single attack against
a PyTorch ResNet-18 model for different epsilons and how to then report
the robust accuracy.
"""
import torch
import numpy as np
from tqdm import tqdm
from scipy import stats
import torchvision.models as models
import eagerpy as ep
from foolbox import PyTorchModel, accuracy, samples
from foolbox.attacks import BoundaryAttack, L2BrendelBethgeAttack, L2PGD, L2CarliniWagnerAttack, L2DeepFoolAttack
from img_utils import get_samples, get_samples_for_cropping
from model_factory import get_model
from tracker import Diary, DiaryPage
from foolbox.criteria import Misclassification


def read_dump(path):
    filepath = 'thesis/{}/raw_data.pkl'.format(path)
    raw = torch.load(open(filepath, 'rb'), map_location='cpu')
    return raw


class Noisy(Misclassification):
    """Considers those perturbed inputs adversarial whose predicted class
    differs from the label.

    Args:
        labels: Tensor with labels of the unperturbed inputs ``(batch,)``.
    """
    def __init__(self, labels, flip_prob, rep):
        super().__init__(labels)
        self.flip_prob = flip_prob
        self.calls = 0
        self.rep = rep
        self.beta = 1

    def __call__(self, perturbed, outputs):
        outputs_, restore_type = ep.astensor_(outputs)
        del perturbed, outputs

        # BAYESIAN
        # logits_ = outputs_.numpy()
        # logits_ = logits_ - np.max(logits_, axis=1, keepdims=True)[0]
        # probs = np.exp(self.beta * logits_)
        # probs = probs / np.sum(probs, axis=1, keepdims=True)
        # probs[probs < 1e-4] = 0
        # prediction = np.zeros(len(probs))
        # for i, prob in enumerate(probs):
        #     prediction[i] = np.random.multinomial(1, prob).argmax()

        # STOCHASTIC
        # classes = outputs_.numpy().argmax(axis=-1)
        # assert classes.shape == self.labels.shape
        # prediction = np.tile(classes.reshape(-1, 1), (1, self.rep))
        # n_samples = self.labels.shape[0]
        # self.calls += n_samples * self.rep
        # n_classes = outputs_.shape[-1]
        # rand_pred = np.random.randint(n_classes-1, size=(n_samples, self.rep))
        # rand_pred[rand_pred == self.labels.numpy()[:,None]] = n_classes - 1
        # indices_to_flip = np.random.rand(n_samples, self.rep) < self.flip_prob
        # prediction[indices_to_flip] = rand_pred[indices_to_flip]
        # prediction = stats.mode(prediction, axis=1)[0].flatten()

        # DETERMINISTIC
        prediction = outputs_.numpy().argmax(axis=-1)
        self.calls += self.labels.shape[0]

        is_adv = ep.astensor(torch.tensor(prediction)) != self.labels
        return restore_type(is_adv)


def search_boundary(x_star, x_t, theta_det, true_label, model):
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


def project(x_star, x_t, label, theta_det, model):
    if len(x_t.shape) == 3:
        x_t = x_t[0]
    probs = model.get_probs(x_t[None])
    if torch.argmax(probs[0]) == label:
        c = 0.25
        x_prev = x_star
        while True:
            x_tt = x_t + c * (x_t - x_star) / np.linalg.norm(x_t - x_star)
            x_tt = np.clip(x_tt, 0, 1)
            if np.max(np.abs(x_tt - x_prev)) == 0:
                break
            x_prev = x_tt
            pred = torch.argmax(model.get_probs(x_tt[None])[0])
            if pred != label:
                x_tt = search_boundary(x_t, x_tt, theta_det, label, model)
                break
            c += c
    else:
        x_tt = search_boundary(x_star, x_t, theta_det, label, model)
    return x_tt


def main() -> None:
    # instantiate a model (could also be a TensorFlow or JAX model)
    det_model = get_model(key='mnist_noman', dataset='mnist', noise='deterministic')
    crop_model = get_model('mnist_noman', 'mnist', noise='cropping', crop_size=22)
    fmodel = PyTorchModel(det_model.model, bounds=(0, 1))
    n_samples = 25
    imgs, lbls = get_samples_for_cropping('mnist', n_samples=n_samples, conf=0.75, model=crop_model)
    if type(imgs) is not np.ndarray:
        imgs = imgs.numpy()
    images, labels = ep.astensors(torch.tensor(imgs[:, None, :, :], dtype=torch.float32), torch.tensor(lbls))
    clean_acc = accuracy(fmodel, images, labels)
    print(f"clean accuracy:  {clean_acc * 100:.1f} %")

    # # apply the attack
    d = 28*28
    theta = 1 / (d*np.sqrt(d))
    flips = [0]
    # BD, MC, VD = {}, {}, {}
    # for rep in [1]:
    #     BD[rep] = {}
    #     VD[rep] = {}
    #     MC[rep] = {}
    #     for flip in flips:
    #         attack = L2DeepFoolAttack()
    #         epsilons = [None]
    #         criterion = Noisy(labels, flip, rep)
    #         raw_advs, clipped_advs, success = attack(fmodel, images, criterion, epsilons=epsilons)
    #         border_distance = torch.zeros(n_samples)
    #         vanilla_distance = torch.zeros(n_samples)
    #         for i, x_t in tqdm(enumerate(raw_advs[0])):
    #             x_t = x_t.numpy()
    #             x_tt = project(imgs[i], x_t, lbls[i], theta, det_model)
    #             border_distance[i] = np.linalg.norm(x_tt - imgs[i]) / np.sqrt(d)
    #             vanilla_distance[i] = np.linalg.norm(x_t - imgs[i]) / np.sqrt(d)
    #         print(f'Flip={flip}, Rep={rep}', end='\t')
    #         print("Border-Distance", np.median(border_distance), end='\t')
    #         print('Model-Calls', criterion.calls)
    #         BD[rep][flip] = border_distance
    #         VD[rep][flip] = vanilla_distance
    #         MC[rep][flip] = criterion.calls
    # torch.save({'BD': BD, 'MC': MC, 'VD': VD}, open('thesis/deepfool_l2.pkl', 'wb'))
    #
    # D = torch.load(open('thesis/deepfool_l2.pkl', 'rb'))
    # BD, MC, VD = D['BD'], D['MC'], D['VD']
    # metric = VD
    # for rep in metric:
    #     for flip in metric[rep]:
    #         n_images = len(metric[rep][flip])
    #         perc_50 = np.median(metric[rep][flip])
    #         perc_40 = np.percentile(metric[rep][flip], 40)
    #         perc_60 = np.percentile(metric[rep][flip], 60)
    #         calls = MC[rep][flip]/n_images
    #         print(f'rep={rep} flip={flip}\t{perc_40}\t{perc_50}\t{perc_60}\t{calls}')
    #
    # raw = read_dump('whitebox_hsj_true_grad')
    raw = read_dump('blackbox_hsj')
    calls = 0
    BD_hsj = np.zeros((n_samples, len(raw[0].iterations)))
    VD_hsj = np.zeros((n_samples, len(raw[0].iterations)))
    for i in tqdm(range(n_samples)):
        diary: Diary = raw[i]
        label = diary.true_label
        x_star = diary.original.numpy()
        for j in range(len(diary.iterations)):
            page: DiaryPage = diary.iterations[j]
            x_t = page.bin_search
            x_tt = project(x_star, x_t.numpy(), label, theta, det_model)
            BD_hsj[i, j] = np.linalg.norm(x_tt - x_star) / np.sqrt(d)
            VD_hsj[i, j] = np.linalg.norm(x_t - x_star) / np.sqrt(d)
        calls += page.calls.bin_search
    metric = VD_hsj
    metric = np.min(metric, axis=1)
    perc_50 = np.median(metric)
    perc_40 = np.percentile(metric, 40)
    perc_60 = np.percentile(metric, 60)
    calls = calls / n_samples
    print(f'\t{perc_40}\t{perc_50}\t{perc_60}\t{calls}')


if __name__ == "__main__":
    main()