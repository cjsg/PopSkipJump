from hopskip import HopSkipJumpAttack
from model_interface import ModelInterface
from adversarial import Adversarial
from model_factory import get_model
from img_utils import get_sample, save_adv_image

dataset = 'mnist'
img, label = get_sample(dataset=dataset, index=0)
img_start, _ = get_sample(dataset=dataset, index=1)
a = Adversarial(image=img, label=label)
a.set_starting_point(img_start, bounds=(0, 1))
models = [get_model(key='mnist', dataset=dataset, bayesian=True)]
model_interface = ModelInterface(models, bounds=(0, 1), sampling_freq=10)
attack = HopSkipJumpAttack(model_interface, a, experiment='mnist_bayes_6to3', dataset='mnist')
b = attack.attack(a, iterations=64)
# save_adv_image(b.perturbed, 'adv/mnist_64_otherclass.png', dataset=dataset)
pass
