import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time


random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


def train_network(network, path, n_epochs, optim_name, batch_size_train, batch_size_test,
                  learning_rate, momentum, log_interval):
    train_loader = torch.utils.data.DataLoader(
      torchvision.datasets.MNIST('data', train=True, download=True,
                                 transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                     (0.1307,), (0.3081,))
                                 ])),
      batch_size=batch_size_train, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
      torchvision.datasets.MNIST('data', train=False, download=True,
                                 transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                     (0.1307,), (0.3081,))
                                 ])),
      batch_size=batch_size_test, shuffle=True)

    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)

    if optim_name == 'sgd':
        optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
    elif optim_name == 'adam':
        optimizer = optim.Adam(network.parameters(), lr=learning_rate)
    else:
        raise RuntimeError


    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

    def train(epoch):
      epoch_time = time.time()
      network.train()
      for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
          # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTime: {:.0f}'.format(
          #   epoch, batch_idx * len(data), len(train_loader.dataset),
          #   100. * batch_idx / len(train_loader), loss.item(), time.time() - epoch_time))
          train_losses.append(loss.item())
          train_counter.append(
            (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
          torch.save(network.state_dict(), f'{path}_model.pth')
          torch.save(optimizer.state_dict(), f'{path}_optimizer.pth')

    def test():
        network.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = network(data)
                test_loss += F.nll_loss(output, target, size_average=False).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)
        accuracy = 100. * correct / len(test_loader.dataset)
        # print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct,
        #             len(test_loader.dataset), accuracy))
        return test_loss, accuracy

    test()
    print(f'n_epochs: {n_epochs}, optim: {optim_name}, lr: {learning_rate}, momentum: {momentum}')
    for epoch in range(1, n_epochs + 1):
        train(epoch)
        test_loss, accuracy = test()
        print(f'\tepoch: {epoch}, test_loss: {test_loss}, test_acc: {accuracy}')
    torch.save(network, f'{path}_network.pth')


def main():
    import numpy as np
    n_epochs = 10
    batch_size_train = 64
    batch_size_test = 1000
    log_interval = 20
    network = Net()
    lrs = np.logspace(-3, -1, 11).round(3)
    momens = np.linspace(0.1, 0.7, 7).round(1)
    for optim_name in ['sgd', 'adam']:
        for lr in lrs:
            if optim_name == 'sgd':
                for momen in momens:
                    path = f'data/model_dumps/mnist_{optim_name}_{lr}_{momen}'
                    train_network(network, path, n_epochs, optim_name, batch_size_train, batch_size_test,
                                  lr, momen, log_interval)
            else:
                momen = 0.5
                path = f'data/model_dumps/mnist_{optim_name}_{lr}_{momen}'
                train_network(network, path, n_epochs, optim_name, batch_size_train, batch_size_test,
                              lr, momen, log_interval)
            break
        break

if __name__ == '__main__':
    main()


