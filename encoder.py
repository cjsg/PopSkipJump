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
        return encodings.view([-1] + self.shape)

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
    if encoder_type == 'identity':
        return Encoder(dataset, target_dim, device)
    elif encoder_type == 'pca':
        return PCAEncoder(dataset, target_dim, device)
    else:
        raise RuntimeError(f'Unknown Encoder Type: {encoder_type}')


def main():
    dataset = 'mnist'
    ticks = [784]
    for d in ticks:
        encoder_id = Encoder(dataset, d, 'cpu')
        encoder_pca = PCAEncoder(dataset, d, 'cpu')

        def printd(a, b, msg):
            a_ = encoder_id.decompress(a)
            b_ = encoder_pca.decompress(b)
            d = torch.norm(a_ - b_)
            print(msg, d)

        # samples = encoder.get_dataset_samples(train=False)
        # Y = samples
        # Z = encoder.compress(Y)
        # Y_ = encoder.decompress(Z)
        from model_factory import get_model
        model = get_model(key='mnist_noman', dataset=dataset)
        diary_id = torch.load(open('thesis/exp_id/raw_data.pkl', 'rb'))[0]
        diary_pca = torch.load(open('thesis/exp_pca/raw_data.pkl', 'rb'))[0]
        # diary_id = torch.load(open('thesis/mnist_1_hsj_isc_3_et_identity_etd_784_r_1_b_1_deterministic_fp_0.00_ns_1/raw_data.pkl', 'rb'))[0]
        # diary_pca= torch.load(open('thesis/mnist_1_hsj_isc_3_et_pca_etd_784_r_1_b_1_deterministic_fp_0.00_ns_1/raw_data.pkl', 'rb'))[0]
        printd(diary_id.original, diary_pca.original, 'original')
        printd(diary_id.initial_image, diary_pca.initial_image, 'initial_image')
        printd(diary_id.initial_projection, diary_pca.initial_projection, 'initial_projection')
        page_id = diary_id.iterations[0]
        page_pca = diary_pca.iterations[0]
        printd(page_id.approx_grad, page_pca.approx_grad, 'approx_grad')
        printd(page_id.bin_search, page_pca.bin_search, 'bin_search')
        # X_id = encoder_id.decompress(Y_id)
        # X_pca = encoder_pca.decompress(Y_pca)
        # prob_id = model.get_probs(X_id)
        # pred_id = prob_id.argmax(dim=1)
        # prob_pca = model.get_probs(X_pca)
        # pred_pca = prob_pca.argmax(dim=1)
        # print(torch.sum(pred_pca == pred_id))
        # print (d, torch.mean(torch.norm(Y.float() - Y_.float(), dim=(1, 2))))
    pass

# if __name__ == '__main__':
#     main()
#     r = torch.randn((100, 28, 28), dtype=torch.float64)
#     encoder = get_encoder('pca', 'mnist', 784, 'cpu')
#     # rt = encoder.compress(r, centered=True)
#     rt = r.view(-1, 784).matmul(encoder.V)
#     # rtt = encoder.decompress(rt, centered=True)
#     rtt = rt.matmul(encoder.V.T).view(-1, 28, 28)
#     print(torch.norm(r[0] - rtt[0]))
#     pass
