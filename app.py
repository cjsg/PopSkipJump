import argparse
import logging
import os
import pickle
import time
from datetime import datetime

from defaultparams import DefaultParams
from hopskip import HopSkipJumpAttack
from img_utils import get_sample, read_image, get_samples, get_shape
from model_factory import get_model
from model_interface import ModelInterface

logging.root.setLevel(logging.WARNING)

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset",
                    help="(Mandatory) supported: mnist, cifar10")
parser.add_argument("-o", "--exp_name", default=None,
                    help="(Optional) path to the output directory")


def validate_args(args):
    try:
        assert args.dataset is not None
    except AssertionError:
        print("Invalid Arguments. try 'python app.py -h'")
        exit()


def create_attack(exp_name, dataset, params):
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
                            beta=params.beta)]
        # models = [get_model(key='mnist_cw', dataset=args.dataset, noise=NOISE, flip_prob=flip_prob)]

    model_interface = ModelInterface(models, bounds=(0, 1), n_classes=10, slack=params.slack, noise=params.noise,
                                     new_adv_def=params.new_adversarial_def)
    return HopSkipJumpAttack(model_interface, get_shape(dataset), experiment=exp_name, params=params)


def run_attack(attack, dataset, params):
    starts = None
    if params.experiment_mode:
        if params.orig_image_conf is not None:
            det_model = get_model(key='mnist_noman', dataset=dataset, noise='deterministic')
            imgs, labels = get_samples(n_samples=params.num_samples, conf=params.orig_image_conf, model=det_model)
        else:
            imgs, labels = get_samples(n_samples=params.num_samples)

    else:
        if params.input_image_path is None or params.input_image_label is None:
            img, label = get_sample(dataset=dataset, index=0)
        else:
            img, label = read_image(params.input_image_path), params.input_image_label
        imgs, labels = [img], [label]

        if params.init_image_path is not None:
            starts = [read_image(params.init_image_path)]
    return attack.attack(imgs, labels, starts, iterations=params.num_iterations, average=params.average,
                         flags=params.flags)


def main(params=None):
    args = parser.parse_args()
    validate_args(args)

    logging.warning(params)
    if params.sampling_freq_approxgrad is None:
        params.sampling_freq_approxgrad = params.sampling_freq_binsearch
    exp_name = args.exp_name if params.experiment_name is None else params.experiment_name
    dataset = args.dataset

    attack = create_attack(exp_name, dataset, params)
    median_distance, additional = run_attack(attack, dataset, params)

    pickle.dump(additional, open('adv/{}/raw_data.pkl'.format(exp_name), 'wb'))
    logging.warning('Saved output at "{}"'.format(exp_name))
    logging.warning('Median_distance: {}'.format(median_distance))


if __name__ == '__main__':
    hyperparams = DefaultParams()
    hyperparams.num_iterations = 32
    hyperparams.noise = 'bayesian'
    # hyperparams.hopskipjumpattack = True
    # hyperparams.remember_all = True
    hyperparams.experiment_name = 'prob'
    hyperparams.num_samples = 1
    # hyperparams.beta = 20
    start = time.time()
    main(params=hyperparams)
    print(time.time() - start)
    pass
