import argparse
import logging
import os
import torch
import time
from datetime import datetime

from defaultparams import DefaultParams
from popskip import PopSkipJump
from hopskip import HopSkipJump
from img_utils import get_sample, read_image, get_samples, get_shape, get_device, find_adversarial_images
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
    assert args.dataset is not None


def create_attack(exp_name, dataset, params):
    exp_path = 'adv/{}'.format(exp_name)
    if os.path.exists(exp_path):
        logging.info("Path: '{}' already exists. Overwriting it!!!".format(exp_name))
    else:
        os.makedirs(exp_path)

    models = [get_model(k, dataset, params.noise, params.flip_prob, params.beta, get_device())
              for k in params.model_keys]
    model_interface = ModelInterface(models, bounds=(0, 1), n_classes=10, slack=params.slack, noise=params.noise,
                                     new_adv_def=params.new_adversarial_def, device=get_device())

    if params.attack == 'hopskip':
        return HopSkipJump(model_interface, get_shape(dataset), params=params, device=get_device())
    elif params.attack == 'popskip':
        return PopSkipJump(model_interface, get_shape(dataset), params=params, device=get_device())
    else:
        raise RuntimeError(f"Attack not found: {params.attack}")


def run_attack(attack, dataset, params):
    starts = None
    if params.experiment_mode:
        det_model = get_model(key=params.model_keys[0], dataset=dataset, noise='deterministic')
        imgs, labels = get_samples(n_samples=params.num_samples, conf=params.orig_image_conf, model=det_model)
        starts = find_adversarial_images(labels)
    else:
        if params.input_image_path is None or params.input_image_label is None:
            img, label = get_sample(dataset=dataset, index=0)
        else:
            img, label = read_image(params.input_image_path), params.input_image_label
        imgs, labels = [img], [label]

        if params.init_image_path is not None:
            starts = [read_image(params.init_image_path)]
    return attack.attack(imgs, labels, starts, iterations=params.num_iterations)


def merge_params(params, args):
    if params.sampling_freq_approxgrad is None:
        params.sampling_freq_approxgrad = params.sampling_freq_binsearch
    params.prior_frac = args.prior_frac
    params.queries = args.queries_per_loc
    params.grad_queries = args.grad_queries
    return params


def get_experiment_name(args, params):
    if params.experiment_name is not None:
        return params.experiment_name
    if args.exp_name is not None:
        return args.exp_name
    return '%s' % datetime.now().strftime("%b%d_%H%M%S")


def main(params=None):
    args = parser.parse_args()
    validate_args(args)
    params = merge_params(params, args)
    exp_name = get_experiment_name(args, params)
    dataset = args.dataset

    attack = create_attack(exp_name, dataset, params)
    median_distance, additional = run_attack(attack, dataset, params)
    torch.save(additional, open('adv/{}/raw_data.pkl'.format(exp_name), 'wb'))
    logging.warning('Saved output at "{}"'.format(exp_name))
    logging.warning('Median_distance: {}'.format(median_distance))
    return median_distance


if __name__ == '__main__':
    hyperparams = DefaultParams()
    hyperparams.num_iterations = 32
    hyperparams.attack = 'popskip'
    # hyperparams.noise = 'deterministic'
    # hyperparams.hopskipjumpattack = True
    hyperparams.experiment_name = 'del_later'
    hyperparams.num_samples = 20
    start = time.time()
    median = main(params=hyperparams)
    print(time.time() - start)
    pass
