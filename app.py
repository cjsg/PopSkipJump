from hopskip import HopSkipJumpAttack
from model_interface import ModelInterface
from adversarial import Adversarial
from model_factory import get_model
from img_utils import get_sample, save_all_images


def main():
    dataset = 'mnist'
    exp_name = 'adv/mnist_bayes_6to3'
    img, label = get_sample(dataset=dataset, index=0)
    img_start, _ = get_sample(dataset=dataset, index=1)
    a = Adversarial(image=img, label=label)
    a.set_starting_point(img_start, bounds=(0, 1))
    models = [get_model(key='mnist', dataset=dataset, bayesian=False)]
    model_interface = ModelInterface(models, bounds=(0, 1), sampling_freq=1)
    attack = HopSkipJumpAttack(model_interface, a, experiment=exp_name, dataset='mnist')
    results = attack.attack(a, iterations=64)
    save_all_images(exp_name, results['iterations'], dataset)


if __name__ == '__main__':
    main()
    pass
