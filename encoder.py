import os
import torch
import torchvision.datasets as datasets
from sklearn.decomposition import PCA

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

    def compress(self, images):
        return images.flatten(start_dim=1).type(torch.float)

    def decompress(self, encodings):
        return encodings.view([-1] + self.shape)

    def get_dataset_samples(self):
        if self.dataset == 'mnist':
            data = datasets.MNIST(root="data", train=True, download=True, transform=None)
            samples = data.data
        else:
            raise RuntimeError(f'Unknown dataset : {self.dataset}')
        return samples


class PCAEncoder(Encoder):
    def __init__(self, dataset, n_components, device):
        super().__init__(dataset, n_components, device)
        encoder_path = f'data/{dataset}_pca_{n_components}.pkl'
        if True or not os.path.exists(encoder_path):
            samples = self.get_dataset_samples()
            X = samples.flatten(start_dim=1).type(torch.float)
            mean_ = torch.mean(X, dim=0)
            U, S, V = torch.pca_lowrank(X - mean_, q=n_components)
            torch.save({'mean': mean_, 'V': V}, open(encoder_path, 'wb'))
            # pca = PCA(n_components=n_components)
            # pca = pca.fit(X)
            # pickle.dump(pca, open(f'data/{dataset}_{self.type}_{n_components}.pkl', 'wb'))
        encoder = torch.load(open(encoder_path, 'rb'), map_location=self.device)
        self.mean = encoder['mean']
        self.V = encoder['V']

    def compress(self, images):
        X = images.flatten(start_dim=1).type(torch.float)
        X -= self.mean
        return X.matmul(self.V)
        # X = self.encoder.transform(images.flatten(start_dim=1))
        # return torch.tensor(X)

    def decompress(self, encodings):
        X = encodings.matmul(self.V.T) + self.mean
        X = torch.clamp(X, self.clip_min, self.clip_max)
        return X.view([-1] + self.shape)
        # X = self.encoder.inverse_transform(encodings)
        # return torch.tensor(X).view([-1] + self.shape)


def get_encoder(encoder_type, dataset, target_dim, device):
    if encoder_type == 'identity':
        return Encoder(dataset, target_dim, device)
    elif encoder_type == 'pca':
        return PCAEncoder(dataset, target_dim, device)
    else:
        raise RuntimeError(f'Unknown Encoder Type: {encoder_type}')

# if __name__ == '__main__':
#     import torchvision.datasets as datasets
#     dataset = 'mnist'
#     encoder = Encoder('pca', dataset, 784)
#     samples = datasets.MNIST(root="data", train=True, download=True, transform=None).data
#     Y = samples[:5]
#     Z = encoder.compress(Y)
#     Y_ = encoder.decompress(Z)
#     print (torch.norm(Y[0] - Y_[0]))
#     pass
