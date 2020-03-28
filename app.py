import numpy as np
from PIL import Image

from cifar10_models import *
from hopskip import HopSkipJumpAttack
from model_interface import ModelInterface
from adversarial import Adversarial

img_pil1 = Image.open('data/cifar10_00_3.png')
img_pil2 = Image.open('data/cifar10_01_8.png')
img_np1 = np.asarray(img_pil1)
img_np2 = np.asarray(img_pil2)
model = densenet121(pretrained=True).eval()
batch = np.stack([img_np1])


def save_adv_image(image, path):
    data = image * 255
    img = Image.fromarray(np.uint8(data), 'RGB')
    img.save(path)


bounds = (0, 1)
if bounds != (0, 255):
    img_np1 = img_np1 / 255 * (bounds[1] - bounds[0]) + bounds[0]
    img_np2 = img_np2 / 255 * (bounds[1] - bounds[0]) + bounds[0]
a = Adversarial(image=img_np2, label=8)
model_interface = ModelInterface(model, bounds=(0, 1))
attack = HopSkipJumpAttack(model_interface, a)
b = attack.attack(a, iterations=10)

save_adv_image(b.perturbed, 'adv10.png')
save_adv_image(b.unperturbed, 'ori.png')

# preprocessing = dict(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010], axis=-3)
pass
