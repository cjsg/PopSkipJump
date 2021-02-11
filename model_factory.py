import torch
from torchvision import transforms
import torch.nn.functional as F
from cifar10_models import *
from pytorchmodels import MNIST_Net, CWMNISTNetwork
from img_utils import show_image


class Model:
    def __init__(self, model, noise=None, n_classes=10, flip_prob=0.25, beta=1.0, device=None, smoothing_noise=0.,
                 crop_size=None):
        self.model = model
        self.noise = noise
        self.n_classes = n_classes
        self.flip_prob = flip_prob
        self.beta = beta
        self.device = device
        self.smoothing_noise = smoothing_noise
        self.crop_size = crop_size

    def predict(self, images):
        images = images.permute(0, 3, 1, 2)
        transform = transforms.Compose([transforms.Normalize([0.4914, 0.4822, 0.4465],
                                                             [0.2023, 0.1994, 0.2010])])
        img_tr = [transform(i) for i in images]
        outs = self.model(torch.stack(img_tr))
        return outs.detach()

    # TODO: Will be deprecated soon (Only one usage in crunch_expermiments.py)
    def ask_model(self, images):
        if self.noise == 'bayesian':
            logits = self.predict(images)
            logits = logits - torch.max(logits, dim=1, keepdim=True)[0]
            probs = torch.exp(self.beta * logits)
            probs = probs / torch.sum(probs, dim=1, keepdim=True)
            probs[probs < 1e-4] = 0
            sample = torch.multinomial(probs, 1)
            return sample.flatten()
        elif self.noise == 'stochastic':
            logits = self.predict(images)
            pred = torch.argmax(logits, dim=1)
            rand = torch.randint(self.n_classes, size=[images.shape[0]])
            flip_prob = torch.rand(len(images))
            pred[flip_prob < self.flip_prob] = rand[flip_prob < self.flip_prob]
            return pred
        elif self.noise == 'smoothing':
            rv = torch.randn(size=images.shape, device=images.device)
            images_ = images + self.smoothing_noise * rv
            images_ = torch.clamp(images_, 0, 1)
            logits = self.predict(images_)
            return torch.argmax(logits, dim=1)
        elif self.noise == 'cropping':
            size = images.shape[1]
            x_start = torch.randint(low=0, high=size + 1 - self.crop_size, size=(1, len(images)))[0]
            x_end = x_start + self.crop_size
            y_start = torch.randint(low=0, high=size + 1 - self.crop_size, size=(1, len(images)))[0]
            y_end = y_start + self.crop_size
            cropped = [b[x_start[i]:x_end[i], y_start[i]:y_end[i]] for i, b in enumerate(images)]
            cropped_batch = torch.stack(cropped)
            if cropped_batch.ndim == 4:
                resized = F.interpolate(cropped_batch.permute(0, 3, 1, 2), size, mode='bilinear')
                resized = resized.permute(0, 2, 3, 1)
            else:
                resized = F.interpolate(cropped_batch.unsqueeze(dim=1), size, mode='bilinear')
                resized = resized.squeeze(dim=1)
            logits = self.predict(resized)
            return torch.argmax(logits, dim=1)
        elif self.noise in ['deterministic', 'dropout']:
            logits = self.predict(images)
            return torch.argmax(logits, dim=1)
        else:
            raise RuntimeError(f'Unknown Noise type: {self.noise}')

    def get_probs(self, images):
        if type(images) != torch.Tensor:
            images = torch.tensor(images, dtype=torch.float32)
        logits = self.predict(images)
        # logits = logits.numpy()
        logits = logits - torch.max(logits, dim=1, keepdim=True)[0]
        probs = torch.exp(self.beta * logits)
        probs = probs / torch.sum(probs, dim=1, keepdim=True)
        # sample = [np.argmax(np.random.multinomial(1, prob)) for prob in probs]
        return probs

    def get_grads(self, images, true_label):
        # TODO: this line will not work for noisy model.
        # wrong_labels = self.ask_model(images)
        if images.ndim == 3:
            images_ = images.unsqueeze(1).type(torch.float32)
        else:
            images_ = images.permute(0, 3, 1, 2)
            transform = transforms.Compose([transforms.Normalize([0.4914, 0.4822, 0.4465],
                                                                 [0.2023, 0.1994, 0.2010])])
            images_ = torch.stack([transform(i) for i in images_])
        t_images = torch.tensor(images_, requires_grad=True, device=self.device)
        t_outs = self.model(t_images)
        grad = torch.zeros(t_images.shape)
        for i in range(len(images)):
            _grad_true = torch.autograd.grad(t_outs[i, true_label], t_images, create_graph=True)[0]
            grad[i] = - _grad_true[i]
        if images.ndim == 3:
            grad = grad[:, 0, :, :]
        else:
            grad = grad.permute(0, 2, 3, 1)
        return grad.detach()


def get_model(key, dataset, noise=None, flip_prob=0.25, beta=1.0, device=None, smoothing_noise=0., crop_size=None,
              drop_rate=0.):
    class MNIST_Model(Model):
        def predict(self, images):
            images = images.unsqueeze(dim=1)
            outs = self.model(images.float())
            return outs.detach()

    if key == 'mnist_cnn':
        pytorch_model = MNIST_Net()
        pytorch_model.load_state_dict(torch.load('mnist_models/mnist_model.pth'))
        pytorch_model.eval()
        if noise == "dropout":
            pytorch_model.conv2_drop.p = drop_rate
            pytorch_model.conv2_drop.train()
        return MNIST_Model(pytorch_model, noise, n_classes=10, flip_prob=flip_prob, beta=beta, device=device,
                           smoothing_noise=smoothing_noise, crop_size=crop_size)
    if key == 'mnist_cw':
        pytorch_model = CWMNISTNetwork()
        pytorch_model.load_state_dict(torch.load('mnist_models/cw_mnist_cnn.pt', map_location='cpu'))
        pytorch_model.eval()
        return MNIST_Model(pytorch_model, noise, n_classes=10, flip_prob=flip_prob)
    if key == 'cifar10':
        if noise == "dropout":
            pytorch_model = densenet121(pretrained=True, drop_rate=drop_rate).eval()
        else:
            pytorch_model = densenet121(pretrained=True, drop_rate=0).eval()
        return Model(pytorch_model, noise, n_classes=10, beta=beta, device=device,
                     smoothing_noise=smoothing_noise, crop_size=crop_size)
    if key == 'human':
        class Human(Model):
            def ask_model(self, images):
                results = list()
                for image in images:
                    show_image(image, dataset=dataset)
                    res = int(input("Whats the class?: ").strip())
                    results.append(res)
                return torch.tensor(results)

        return Human(model=None)
