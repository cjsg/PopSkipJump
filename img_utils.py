import numpy as np
import torchvision.datasets as datasets
from PIL import Image

CIFAR_PATHS = ['cifar10_00_3.png', 'cifar10_01_8.png']
MNIST_PATHS = ['mnist_01_3.jpg', 'mnist_02_6.jpg', 'mnist_03_0.jpg', 'mnist_04_7.jpg', 'mnist_05_9.jpg']


def read_image(path):
    return np.asarray(Image.open(path)) / 255


def get_sample(dataset, index=0):
    if dataset == 'cifar10':
        filename = CIFAR_PATHS[index]
    elif dataset == 'mnist':
        filename = MNIST_PATHS[index]
    else:
        raise
    img = read_image('data/{}'.format(filename))
    label = int(filename.split('.')[0].split('_')[-1])
    return img, label


def get_shape(dataset):
    if dataset == 'cifar10':
        return 32, 32, 3
    if dataset == 'mnist':
        return 28, 28
    raise RuntimeError("Unknown Dataset: {}".format(dataset))


def get_samples(n_samples=16):
    np.random.seed(42)
    test_data = datasets.MNIST(root="data", train=False, download=True, transform=None)
    indices = np.random.choice(len(test_data), n_samples, replace=False)
    images = test_data.data[indices].numpy() / 255.0
    labels = test_data.test_labels[indices].numpy()
    return images, labels


def save_adv_image(image, path, dataset='cifar10'):
    data = image * 255
    if dataset == 'mnist':
        img = Image.fromarray(np.uint8(data), 'L')
    else:
        img = Image.fromarray(np.uint8(data), 'RGB')
    img.save(path)


def show_image(image, dataset='cifar10'):
    data = image * 255
    if dataset == 'mnist':
        img = Image.fromarray(np.uint8(data), 'L')
    else:
        img = Image.fromarray(np.uint8(data), 'RGB')
    img.show()


def get_concat_h(im1, im2):
    dst = Image.new('L', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


def get_concat_v(im1, im2):
    dst = Image.new('L', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst


def one_big_image(exp_name):
    rows = []
    for row in range(0, 8):
        h = Image.open('{}/{}.png'.format(exp_name, 8 * row + 1))
        for i in range(2, 9):
            img = Image.open('{}/{}.png'.format(exp_name, 8 * row + i))
            h = get_concat_h(h, img)
        rows.append(h)
    cur = rows[0]
    for row in rows[1:]:
        cur = get_concat_v(cur, row)
    cur.save('{}/combined.png'.format(exp_name))


def save_all_images(exp_name, images, dataset):
    for i, image in enumerate(images):
        save_adv_image(image, "%s/%d.png" % (exp_name, i + 1), dataset)
    one_big_image(exp_name)
