from hopskip import HopSkipJumpAttack
from model_interface import ModelInterface
from adversarial import Adversarial
from model_factory import get_model
from img_utils import get_sample, save_all_images, read_image
from conf import *
import logging
from datetime import datetime
import os

logging.root.setLevel(logging.INFO)


def validate_args(args):
    try:
        assert args.dataset is not None
    except:
        print("Invalid Arguments. try 'python app.py -h'")
        exit()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset",
                        help="(Mandatory) supported: mnist, cifar10")
    parser.add_argument("-o", "--output", default=None,
                        help="(Optional) path to the output directory")
    args = parser.parse_args()
    validate_args(args)
    exp_name = args.output
    if args.output is None:
        exp_name = 'adv/%s' % datetime.now().strftime("%b%d_%H%M%S")

    if os.path.exists(exp_name):
        logging.info("Path: '{}' already exists. Overwriting it!!!".format(exp_name))
    else:
        os.makedirs(exp_name)

    if ATTACK_INPUT_IMAGE is None or ATTACK_INPUT_LABEL is None:
        img, label = get_sample(dataset=args.dataset, index=0)
    else:
        img, label = read_image(ATTACK_INPUT_IMAGE), ATTACK_INPUT_LABEL
    a = Adversarial(image=img, label=label)

    # img_start, _ = get_sample(dataset=args.dataset, index=1)
    if ATTACK_INITIALISE_IMAGE is not None:
        img_start = read_image(ATTACK_INITIALISE_IMAGE)
        a.set_starting_point(img_start, bounds=(0, 1))

    # For now choice of model is fixed for a particular dataset
    if ASK_HUMAN:
        models = [get_model(key='human', dataset=args.dataset, bayesian=BAYESIAN)]
    else:
        models = [get_model(key='mnist', dataset=args.dataset, bayesian=BAYESIAN)]

    model_interface = ModelInterface(models, bounds=(0, 1))
    attack = HopSkipJumpAttack(model_interface, a, experiment=exp_name, dataset=args.dataset)
    results = attack.attack(a, iterations=NUM_ITERATIONS)
    save_all_images(exp_name, results['iterations'], args.dataset)
    logging.info('Saved output images at "{}"'.format(exp_name))


if __name__ == '__main__':
    main()
    pass
