from hopskip import HopSkipJumpAttack
from model_interface import ModelInterface
from adversarial import Adversarial
from model_factory import get_model
from img_utils import get_sample, save_all_images
from conf import BAYESIAN, NUM_ITERATIONS
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
    parser.add_argument("-d", "--dataset", help="(Mandatory) supported: mnist, cifar10")
    parser.add_argument("-o", "--output", default=None, help="path to the output directory")
    args = parser.parse_args()
    validate_args(args)
    exp_name = args.output
    if args.output is None:
        exp_name = 'adv/%s' % datetime.now().strftime("%b%d_%H%M%S")

    if os.path.exists(exp_name):
        logging.info("Path: '{}' already exists. Overwriting it!!!".format(exp_name))
    else:
        os.makedirs(exp_name)

    img, label = get_sample(dataset=args.dataset, index=0)
    img_start, _ = get_sample(dataset=args.dataset, index=1)
    a = Adversarial(image=img, label=label)
    a.set_starting_point(img_start, bounds=(0, 1))
    models = [get_model(key='mnist', dataset=args.dataset, bayesian=BAYESIAN)]
    model_interface = ModelInterface(models, bounds=(0, 1))
    attack = HopSkipJumpAttack(model_interface, a, experiment=exp_name, dataset=args.dataset)
    results = attack.attack(a, iterations=NUM_ITERATIONS)
    save_all_images(exp_name, results['iterations'], args.dataset)
    logging.info('Saved output images at "{}"'.format(exp_name))


if __name__ == '__main__':
    main()
    pass
