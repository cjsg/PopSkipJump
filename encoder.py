import os
import torch
import torchvision.datasets as datasets

shapes = {'mnist': [28, 28]}


class Encoder(object):
    def __init__(self, dataset, n_components, device):
        self.dataset = dataset
        self.shape = shapes[dataset]
        self.n_components = n_components
        self.device = device
        self.clip_min, self.clip_max = 0, 1
        if not os.path.exists('./data'):
            os.makedirs('./data')

    def compress(self, images, centered=False):
        return images.flatten(start_dim=1).type(torch.float64)

    def decompress(self, encodings, centered=False):
        X = encodings.view([-1] + self.shape)
        X = torch.clamp(X, self.clip_min, self.clip_max)
        return X

    def get_dataset_samples(self, train=True):
        if self.dataset == 'mnist':
            data = datasets.MNIST(root="data", train=train, download=True, transform=None)
            samples = data.data
        else:
            raise RuntimeError(f'Unknown dataset : {self.dataset}')
        return samples.type(torch.float64) / 255.0


class PCAEncoder(Encoder):
    def __init__(self, dataset, n_components, device):
        super().__init__(dataset, n_components, device)
        encoder_path = f'data/{dataset}_pca_{n_components}.pkl'
        # if True:
        if not os.path.exists(encoder_path):
            samples = self.get_dataset_samples()
            X = samples.flatten(start_dim=1)
            mean_ = torch.mean(X, dim=0)
            U, S, V = torch.pca_lowrank(X - mean_, q=n_components)
            torch.save({'mean': mean_, 'V': V}, open(encoder_path, 'wb'))
        encoder = torch.load(open(encoder_path, 'rb'), map_location=self.device)
        self.mean = encoder['mean']
        self.V = encoder['V']

    def compress(self, images, centered=False):
        assert (images.dtype == torch.float64)
        X = images.flatten(start_dim=1)
        if not centered:
            X = X - self.mean
        return X.matmul(self.V)

    def decompress(self, encodings, centered=False):
        X = encodings.matmul(self.V.T)
        if not centered:
            X = X + self.mean
        X = torch.clamp(X, self.clip_min, self.clip_max)
        return X.view([-1] + self.shape)


def get_encoder(encoder_type, dataset, target_dim, device):
    if encoder_type in ['identity', 'vanilla']:
        return Encoder(dataset, target_dim, device)
    elif encoder_type == 'pca':
        return PCAEncoder(dataset, target_dim, device)
    else:
        raise RuntimeError(f'Unknown Encoder Type: {encoder_type}')


def main():
    dataset = 'mnist'
    ticks = [50, 100, 150, 200, 300, 400, 500, 600, 700]
    for d in ticks:
        encoder_id = Encoder(dataset, d, 'cpu')
        encoder_pca = PCAEncoder(dataset, d, 'cpu')
        encoder = encoder_pca
        def printd(a, b, msg):
            a_ = encoder_id.decompress(a)
            b_ = encoder_pca.decompress(b)
            d = torch.norm(a_ - b_)
            print(msg, d)

        samples = encoder.get_dataset_samples(train=False)
        Y = samples
        Z = encoder.compress(Y)
        Y_ = encoder.decompress(Z)
        print(d, torch.mean(torch.norm(Y.float() - Y_.float(), dim=(1, 2))))
    pass


if __name__ == '__main__':
    main()
#     r = torch.randn((100, 28, 28), dtype=torch.float64)
#     encoder = get_encoder('pca', 'mnist', 784, 'cpu')
#     # rt = encoder.compress(r, centered=True)
#     rt = r.view(-1, 784).matmul(encoder.V)
#     # rtt = encoder.decompress(rt, centered=True)
#     rtt = rt.matmul(encoder.V.T).view(-1, 28, 28)
#     print(torch.norm(r[0] - rtt[0]))
#     pass
