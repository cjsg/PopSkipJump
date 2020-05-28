import torch
import numpy as np
from torchvision import transforms
from cifar10_models import *
from pytorchmodels import MNIST_Net
from img_utils import show_image


class Model:
    def __init__(self, model, noise=None, n_classes=10, flip_prob=0.25):
        self.model = model
        self.noise = noise
        self.n_classes = n_classes
        self.flip_prob = flip_prob

    def predict(self, images):
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize([0.4914, 0.4822, 0.4465],
                                                             [0.2023, 0.1994, 0.2010])])
        img_tr = [transform(i) for i in images]
        outs = self.model(torch.stack(img_tr))
        return outs.detach().numpy()

    def ask_model(self, images):
        logits = self.predict(images)
        if self.noise == 'bayesian':
            logits = logits - np.max(logits, axis=1, keepdims=True)
            probs = np.exp(logits)
            probs = probs / np.sum(probs, axis=1, keepdims=True)
            sample = [np.argmax(np.random.multinomial(1, prob)) for prob in probs]
            return np.array(sample)
        elif self.noise == 'stochastic':
            pred = np.argmax(logits, axis=1)
            rand = np.random.randint(self.n_classes, size=len(images))
            flip_prob = np.random.uniform(0, 1, len(images))
            pred[flip_prob < self.flip_prob] = rand[flip_prob < self.flip_prob]
            return pred
        else:
            return np.argmax(logits, axis=1)

    def get_probs(self, images):
        logits = self.predict(images)
        logits = logits - np.max(logits, axis=1, keepdims=True)
        probs = np.exp(logits)
        probs = probs / np.sum(probs, axis=1, keepdims=True)
        # sample = [np.argmax(np.random.multinomial(1, prob)) for prob in probs]
        return np.array(probs)


def get_model(key, dataset, noise=None, flip_prob=0.25):
    if key == 'mnist':
        class MNIST_Model(Model):
            def predict(self, images):
                images = np.expand_dims(images, axis=1).astype(np.float32)
                outs = self.model(torch.tensor(images))
                return outs.detach().numpy()

        pytorch_model = MNIST_Net()
        pytorch_model.load_state_dict(torch.load('mnist_models/mnist_model.pth'))
        pytorch_model.eval()
        return MNIST_Model(pytorch_model, noise, n_classes=10, flip_prob=flip_prob)
    if key == 'cifar10':
        return Model(densenet121(pretrained=True).eval(), noise, n_classes=10)
    if key == 'human':
        class Human(Model):
            def ask_model(self, images):
                results = list()
                for image in images:
                    show_image(image, dataset=dataset)
                    res = int(input("Whats the class?: ").strip())
                    results.append(res)
                return np.array(results)

        return Human(model=None)
