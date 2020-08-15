from hopskip import HopSkipJumpAttack
from model_interface import ModelInterface
import torchvision.datasets as datasets
from model_factory import get_model
from img_utils import get_sample, read_image
import logging
from defaultparams import DefaultParams
from datetime import datetime
import time
import os
import numpy as np
import argparse
import pickle

logging.root.setLevel(logging.DEBUG)


def get_samples(n_samples=16):
    np.random.seed(42)
    test_data = datasets.MNIST(root="data", train=False, download=True, transform=None)
    indices = np.random.choice(len(test_data), n_samples, replace=False)
    images = test_data.data[indices].numpy() / 255.0
    labels = test_data.test_labels[indices].numpy()
    return images, labels


def validate_args(args):
    try:
        assert args.dataset is not None
    except:
        print("Invalid Arguments. try 'python app.py -h'")
        exit()


def main(params=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset",
                        help="(Mandatory) supported: mnist, cifar10")
    parser.add_argument("-o", "--exp_name", default=None,
                        help="(Optional) path to the output directory")
    args = parser.parse_args()
    validate_args(args)
    logging.warning(params)
    exp_name = args.exp_name if params.experiment_name is None else params.experiment_name
    if exp_name is None:
        exp_name = 'adv/%s' % datetime.now().strftime("%b%d_%H%M%S")

    if os.path.exists(exp_name):
        logging.info("Path: '{}' already exists. Overwriting it!!!".format(exp_name))
    else:
        os.makedirs(exp_name)

    if params.sampling_freq_approxgrad is None:
        params.sampling_freq_approxgrad = params.sampling_freq_binsearch

    starts = None
    if params.experiment_mode:
        imgs, labels = get_samples(n_samples=params.num_samples)
    else:
        if params.input_image_path is None or params.input_image_label is None:
            img, label = get_sample(dataset=args.dataset, index=0)
        else:
            img, label = read_image(params.input_image_path), params.input_image_label
        imgs, labels = [img], [label]

        if params.init_image_path is not None:
            starts = [read_image(params.init_image_path)]

    # For now choice of model is fixed for a particular dataset
    if params.ask_human:
        models = [get_model(key='human', dataset=args.dataset, noise=params.noise)]
    else:
        models = [get_model(key='mnist_noman', dataset=args.dataset, noise=params.noise, flip_prob=params.flip_prob)]
        # models = [get_model(key='mnist_cw', dataset=args.dataset, noise=NOISE, flip_prob=flip_prob)]

    model_interface = ModelInterface(models, bounds=(0, 1), n_classes=10, slack=params.slack, noise=params.noise)
    attack = HopSkipJumpAttack(model_interface, imgs[0].shape, experiment=exp_name, params=params)
    median_distance, additional = attack.attack(imgs, labels, starts, iterations=params.num_iterations,
                                                average=params.average, flags=params.flags)
    pickle.dump(additional, open('{}/raw_data.pkl'.format(exp_name), 'wb'))
    logging.warning('Saved output at "{}"'.format(exp_name))
    logging.warning('Median_distance: {}'.format(median_distance))


if __name__ == '__main__':
    hyperparams = DefaultParams()
    hyperparams.sampling_freq_binsearch = 32
    hyperparams.num_iterations = 30
    hyperparams.experiment_name = 'del_later'
    start = time.time()
    main(params=hyperparams)
    print(time.time() - start)
    pass
