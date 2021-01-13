import argparse
import logging
import os
import torch
import time
from datetime import datetime
from defaultparams import DefaultParams
from popskip import PopSkipJump, PopSkipJumpTrueLogits
from hopskip import HopSkipJump, HopSkipJumpRepeated, HopSkipJumpRepeatedWithPSJDelta, HopSkipJumpTrueGradient, HopSkipJumpAllGradient
from img_utils import get_sample, read_image, get_samples, get_shape, get_device, find_adversarial_images
from model_factory import get_model
from model_interface import ModelInterface

logging.root.setLevel(logging.WARNING)
OUT_DIR = 'thesis'
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", type=str,
                    help="(Mandatory) supported: mnist, cifar10")
parser.add_argument("-n", "--noise", type=str,
                    help="(Mandatory) supported: deterministic, bayesian")
parser.add_argument("-a", "--attack", type=str,
                    help="(Mandatory) supported: psj, hsj, hsj_rep")
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
parser.add_argument("-r", "--hsja_repeat_queries", type=int, default=1,
                    help="(Optional) how many queries to compute per "
                    "point in HSJ Attack")
parser.add_argument("-ns", "--num_samples", type=int, default=1,
                    help="(Optional) Number of images to attack")
parser.add_argument("-b", "--beta", type=float, default=1,
                    help="(Optional) Beta parameter used in Gibbs Distribution")
parser.add_argument("-sf", "--samples_from", type=int, default=0,
                    help="(Optional) Number of images to skip during sampling")
parser.add_argument("-fp", "--flip_prob", type=float, default=0,
                    help="(Optional) Probability of flipping the outcome of noisy classifier")
parser.add_argument("-tf", "--theta_fac", type=float, default=-1,
                    help="(Optional) Multiplies theta of HSJ with tf")
parser.add_argument("-isc", "--infomax_stop_criteria", type=str, default="estimate_fluctuation",
                    help="(Optional) Stopping Criteria to use in Infomax procedure")
parser.add_argument("-dm", "--distance", type=str, default="L2",
                    help="(Optional) Distance metric for attack. ex L2, Linf")
parser.add_argument("-ef", "--eval_factor", type=float, default=1,
                    help="(Optional) Distance metric for attack. ex L2, Linf")


def validate_args(args):
    assert args.dataset is not None
    assert args.noise is not None
    assert args.attack is not None


def create_attack(exp_name, dataset, params):
    exp_path = '{}/{}'.format(OUT_DIR, exp_name)
    if os.path.exists(exp_path):
        logging.info("Path: '{}' already exists. Overwriting it!!!".format(exp_name))
    else:
        os.makedirs(exp_path)

    models = [get_model(k, dataset, params.noise, params.flip_prob, params.beta, get_device())
              for k in params.model_keys[dataset]]
    model_interface = ModelInterface(models, bounds=params.bounds, n_classes=10, slack=params.slack,
                                     noise=params.noise, device=get_device(), flip_prob=params.flip_prob)
    attacks_factory = {
        'hsj': HopSkipJump,
        'hsj_rep': HopSkipJumpRepeated,
        'hsj_rep_psj_delta': HopSkipJumpRepeatedWithPSJDelta,
        'hsj_true_grad': HopSkipJumpTrueGradient,
        'hsj_all_grad': HopSkipJumpAllGradient,
        'psj': PopSkipJump,
        'psj_true_logits': PopSkipJumpTrueLogits
    }
    return attacks_factory.get(params.attack)(model_interface, get_shape(dataset), get_device(), params)


def run_attack(attack, dataset, params):
    starts = None
    if params.experiment_mode:
        det_model = get_model(key=params.model_keys[dataset][0], dataset=dataset, noise='deterministic')
        imgs, labels = get_samples(dataset, n_samples=params.num_samples, conf=params.orig_image_conf,
                                   model=det_model, samples_from=params.samples_from)
        starts = find_adversarial_images(dataset, labels)
    else:
        if params.input_image_path is None or params.input_image_label is None:
            img, label = get_sample(dataset=dataset, index=0)
        else:
            img, label = read_image(params.input_image_path), params.input_image_label
        imgs, labels = [img], [label]

        if params.init_image_path is not None:
            starts = [read_image(params.init_image_path)]
    return attack.attack(imgs, labels, starts, iterations=params.num_iterations)


def merge_params(params: DefaultParams, args):
    params.noise = args.noise
    params.beta = args.beta
    params.attack = args.attack
    params.dataset = args.dataset
    params.prior_frac = args.prior_frac
    params.queries = args.queries_per_loc
    params.grad_queries = args.grad_queries
    params.hsja_repeat_queries = args.hsja_repeat_queries
    params.num_samples = args.num_samples
    params.samples_from = args.samples_from
    params.flip_prob = args.flip_prob
    params.theta_fac = args.theta_fac
    params.infomax_stop_criteria = args.infomax_stop_criteria
    params.distance = args.distance
    params.eval_factor = args.eval_factor
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
    torch.save(additional, open('{}/{}/raw_data.pkl'.format(OUT_DIR, exp_name), 'wb'))
    logging.warning('Saved output at "{}"'.format(exp_name))
    logging.warning('Median_distance: {}'.format(median_distance))
    return median_distance


if __name__ == '__main__':
    hyperparams = DefaultParams()
    start = time.time()
    median = main(params=hyperparams)
    print(time.time() - start)
    pass
