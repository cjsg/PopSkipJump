import argparse
import logging
import os
import torch
import time
from datetime import datetime

from defaultparams import DefaultParams
# from hopskip import HopSkipJumpAttack
from popskip import PopSkipJump
from hopskip import HopSkipJump
# from our_attack import OurAttack
from img_utils import get_sample, read_image, get_samples, get_shape, get_device
from model_factory import get_model
from model_interface import ModelInterface

logging.root.setLevel(logging.WARNING)

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", type=str,
                    help="(Mandatory) supported: mnist, cifar10")
parser.add_argument("-o", "--exp_name", type=str, default=None,
                    help="(Optional) path to the output directory")
parser.add_argument("-pf", "--prior_frac", type=float, default=1.,
                    help="(Optional) how much to reduce the bin-search "
                    "interval after first round of bin-search")
parser.add_argument("-q", "--queries_per_loc", type=int, default=1,
                    help="(Optional) how many queries to compute per "
                    "Bayesian optimization step in bin-search.")
parser.add_argument("-gq", "--grad_queries", type=int, default=1,
                    help="(Optional) how many queries to compute per "
                    "point in Gradient Approximation step.")


def validate_args(args):
    try:
        assert args.dataset is not None
    except AssertionError:
        print("Invalid Arguments. try 'python app.py -h'")
        exit()


def create_attack(exp_name, dataset, params, prior_frac, queries, grad_queries):
    if exp_name is None:
        exp_name = '%s' % datetime.now().strftime("%b%d_%H%M%S")

    exp_path = 'adv/{}'.format(exp_name)
    if os.path.exists(exp_path):
        logging.info("Path: '{}' already exists. Overwriting it!!!".format(exp_name))
    else:
        os.makedirs(exp_path)

    # Register all the models
    if params.ask_human:
        models = [get_model(key='human', dataset=dataset, noise=params.noise)]
    else:
        models = [get_model(key='mnist_noman', dataset=dataset, noise=params.noise, flip_prob=params.flip_prob,
                            beta=params.beta, device=get_device())]
        # models = [get_model(key='mnist_cw', dataset=args.dataset, noise=NOISE, flip_prob=flip_prob)]

    model_interface = ModelInterface(
        models, bounds=(0, 1), n_classes=10, slack=params.slack,
        noise=params.noise, new_adv_def=params.new_adversarial_def,
        device=get_device())

    # return HopSkipJump(model_interface, get_shape(dataset), params=params, device=get_device(),
    #                          prior_frac=prior_frac, queries=queries)
    return PopSkipJump(model_interface, get_shape(dataset), params=params, device=get_device(),
                             prior_frac=prior_frac, queries=queries, grad_queries=grad_queries)
    # return OurAttack(model_interface, get_shape(dataset), experiment=exp_name, params=params)


def run_attack(attack, dataset, params):
    starts = None
    if params.experiment_mode:
        if params.orig_image_conf is not None:
            det_model = get_model(key='mnist_noman', dataset=dataset, noise='deterministic')
            imgs, labels = get_samples(n_samples=params.num_samples, conf=params.orig_image_conf, model=det_model)
        else:
            imgs, labels = get_samples(n_samples=params.num_samples)
        ii, ll = get_samples(n_samples=10)
        cand_img, cand_lbl = [], []
        for i, l in enumerate(ll):
            if l != ll[0]:
                cand_img = [ii[0], ii[i]]
                cand_lbl = [ll[0], ll[i]]
        starts = []
        for l in labels:
            if l != cand_lbl[0]:
                starts.append(ii[0])
            else:
                starts.append(ii[1])
    else:
        if params.input_image_path is None or params.input_image_label is None:
            img, label = get_sample(dataset=dataset, index=0)
        else:
            img, label = read_image(params.input_image_path), params.input_image_label
        imgs, labels = [img], [label]

        if params.init_image_path is not None:
            starts = [read_image(params.init_image_path)]
    return attack.attack(imgs, labels, starts, iterations=params.num_iterations, average=params.average)


def main(params=None):
    args = parser.parse_args()
    validate_args(args)

    logging.warning(params)
    if params.sampling_freq_approxgrad is None:
        params.sampling_freq_approxgrad = params.sampling_freq_binsearch
    exp_name = args.exp_name if params.experiment_name is None else params.experiment_name
    dataset = args.dataset

    attack = create_attack(
        exp_name, dataset, params, args.prior_frac, args.queries_per_loc, args.grad_queries)
    median_distance, additional = run_attack(attack, dataset, params)
    torch.save(additional, open('adv/{}/raw_data.pkl'.format(exp_name), 'wb'))
    logging.warning('Saved output at "{}"'.format(exp_name))
    logging.warning('Median_distance: {}'.format(median_distance))
    return median_distance


if __name__ == '__main__':
    hyperparams = DefaultParams()
    hyperparams.num_iterations = 3
    # hyperparams.noise = 'deterministic'
    # hyperparams.hopskipjumpattack = True
    hyperparams.experiment_name = 'del_later'
    hyperparams.num_samples = 1
    start = time.time()
    median = main(params=hyperparams)
    # assert 0.002 <= median <= 0.008
    print(time.time() - start)
    pass
