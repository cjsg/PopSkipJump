import numpy as np
from PIL import Image

CIFAR_PATHS = ['cifar10_00_3.png', 'cifar10_01_8.png']
MNIST_PATHS = ['mnist_01_3.jpg', 'mnist_02_6.jpg', 'mnist_03_0.jpg', 'mnist_04_7.jpg', 'mnist_05_9.jpg']


def get_sample(dataset, index=0):
    if dataset == 'cifar10':
        filename = CIFAR_PATHS[index]
    elif dataset == 'mnist':
        filename = MNIST_PATHS[index]
    else:
        raise
    img = np.asarray(Image.open('data/%s' % filename)) / 255
    label = int(filename.split('.')[0].split('_')[-1])
    return img, label


def save_adv_image(image, path, dataset='cifar10'):
    data = image * 255
    if dataset == 'mnist':
        img = Image.fromarray(np.uint8(data), 'L')
    else:
        img = Image.fromarray(np.uint8(data), 'RGB')
    img.save(path)
