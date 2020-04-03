from hopskip import HopSkipJumpAttack
from model_interface import ModelInterface
from adversarial import Adversarial
from model_factory import get_model
from img_utils import get_sample, save_adv_image

dataset = 'mnist'
img, label = get_sample(dataset=dataset, index=0)
a = Adversarial(image=img, label=label)
model_interface = ModelInterface(get_model(key=dataset), bounds=(0, 1))
attack = HopSkipJumpAttack(model_interface, a)
b = attack.attack(a, iterations=4)
save_adv_image(b.perturbed, 'adv/mnist_4.png', dataset=dataset)
pass
