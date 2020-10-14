import torch
from torchvision import transforms
from cifar10_models import *
from pytorchmodels import MNIST_Net, CWMNISTNetwork
from img_utils import show_image


class Model:
    def __init__(self, model, noise=None, n_classes=10, flip_prob=0.25, beta=1.0, device=None):
        self.model = model
        self.noise = noise
        self.n_classes = n_classes
        self.flip_prob = flip_prob
        self.beta = beta
        self.device = device

    def predict(self, images):
        images = images.permute(0, 3, 1, 2)
        transform = transforms.Compose([
                                        transforms.Normalize([0.4914, 0.4822, 0.4465],
                                                             [0.2023, 0.1994, 0.2010])])
        for i in range(images.shape[0]):
            images[i] = transform(images[i])
        outs = self.model(images)
        return outs.detach()

    def ask_model(self, images):
        logits = self.predict(images)
        if self.noise == 'bayesian':
            logits = logits - torch.max(logits, dim=1, keepdim=True)[0]
            probs = torch.exp(self.beta*logits)
            probs = probs / torch.sum(probs, dim=1, keepdim=True)
            probs[probs < 1e-4] = 0
            sample = torch.multinomial(probs, 1)
            return sample.flatten()
        elif self.noise == 'stochastic':
            pred = torch.argmax(logits, dim=1)
            rand = torch.randint(self.n_classes, size=[images.shape[0]])
            flip_prob = torch.rand(len(images))
            pred[flip_prob < self.flip_prob] = rand[flip_prob < self.flip_prob]
            return pred
        else:
            return torch.argmax(logits, dim=1)

    def get_probs(self, images):
        if type(images) != torch.Tensor:
            images = torch.tensor(images, dtype=torch.float32)
        logits = self.predict(images)
        # logits = logits.numpy()
        logits = logits - torch.max(logits, dim=1, keepdim=True)[0]
        probs = torch.exp(self.beta*logits)
        probs = probs / torch.sum(probs, dim=1, keepdim=True)
        # sample = [np.argmax(np.random.multinomial(1, prob)) for prob in probs]
        return probs

    def get_grads(self, images, true_label):
        # TODO: this line will not work for noisy model.
        wrong_labels = self.ask_model(images)
        images = images.unsqueeze(1).type(torch.float32)
        t_images = torch.tensor(images, requires_grad=True)
        t_outs = self.model(t_images)
        grad = torch.zeros(t_images.shape)
        for i in range(len(images)):
            _grad_true = torch.autograd.grad(t_outs[i, true_label], t_images, create_graph=True)[0]
            _grad_wrong = torch.autograd.grad(t_outs[i, wrong_labels[i]], t_images, create_graph=True)[0]
            grad[i] = _grad_true[i] - _grad_wrong[i]
        return grad.detach().numpy()


def get_model(key, dataset, noise=None, flip_prob=0.25, beta=1.0, device=None):
    class MNIST_Model(Model):
        def predict(self, images):
            images = images.unsqueeze(dim=1)
            outs = self.model(images.float())
            return outs.detach()
    if key == 'mnist_noman':
        pytorch_model = MNIST_Net()
        pytorch_model.load_state_dict(torch.load('mnist_models/mnist_model.pth'))
        pytorch_model.eval()
        return MNIST_Model(pytorch_model, noise, n_classes=10, flip_prob=flip_prob, beta=beta, device=device)
    if key == 'mnist_cw':
        pytorch_model = CWMNISTNetwork()
        pytorch_model.load_state_dict(torch.load('mnist_models/cw_mnist_cnn.pt', map_location='cpu'))
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
                return torch.tensor(results)

        return Human(model=None)
