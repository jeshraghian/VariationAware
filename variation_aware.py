from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import quantization_aware as quant


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Visualisation options have been removed. Refer to the notebook to enable.

class NetMem(nn.Module):
    def __init__(self, mnist=True):

        super(NetMem, self).__init__()
        if mnist:
            num_channels = 1
        else:
            num_channels = 3

        self.conv1 = nn.Conv2d(num_channels, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        if mnist:
            self.fc1 = nn.Linear(4 * 4 * 50, 500)
            self.flatten_shape = 4 * 4 * 50
        else:
            self.fc1 = nn.Linear(1250, 500)
            self.flatten_shape = 1250

        self.fc2 = nn.Linear(500, 10)

    def forward(self, x, vis=False, axs=None):
        X = 0
        y = 0

        if vis:
            axs[X, y].set_xlabel('Entry into network, input distribution visualised below: ')
            visualise(x, axs[X, y])

            axs[X, y + 1].set_xlabel("Visualising weights of conv 1 layer: ")
            visualise(self.conv1.weight.data, axs[X, y + 1])

        memristor_dist1 = torch.normal(1, std, size=self.conv1.weight.data.size())
        x = F.relu(self.conv1(x) * memristor_dist1)

        if vis:
            axs[X, y + 2].set_xlabel('Output after conv1 visualised below: ')
            visualise(x, axs[X, y + 2])

            axs[X, y + 3].set_xlabel("Visualising weights of conv 2 layer: ")
            visualise(self.conv2.weight.data, axs[X, y + 3])

        x = F.max_pool2d(x, 2, 2)

        memristor_dist2 = torch.normal(1, std, size=self.conv2.weight.data.size())
        x = F.relu(self.conv2(x) * memristor_dist2)

        if vis:
            axs[X, y + 4].set_xlabel('Output after conv2 visualised below: ')
            visualise(x, axs[X, y + 4])

            axs[X + 1, y].set_xlabel("Visualising weights of fc 1 layer: ")
            visualise(self.fc1.weight.data, axs[X + 1, y])

        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, self.flatten_shape)

        memristor_dist3 = torch.normal(1, std, size=self.fc1.weight.data.size())
        x = F.relu(self.fc1(x) * memristor_dist3)

        if vis:
            axs[X + 1, y + 1].set_xlabel('Output after fc1 visualised below: ')
            visualise(x, axs[X + 1, y + 1])

            axs[X + 1, y + 2].set_xlabel("Visualising weights of fc 2 layer: ")
            visualise(self.fc2.weight.data, axs[X + 1, y + 2])

        memristor_dist4 = torch.normal(1, std, size=self.fc2.weight.data.size())
        x = self.fc2(x) * memristor_dist4

        if vis:
            axs[X + 1, y + 3].set_xlabel('Output after fc2 visualised below: ')
            visualise(x, axs[X + 1, y + 3])

        return F.log_softmax(x, dim=1)


def quantAwareTrainingForward(model, x, stats, std=0, vis=False, axs=None, sym=False, num_bits=1, act_quant=False):
    conv1weight = model.conv1.weight.data
    model.conv1.weight.data = quant.FakeQuantOp.apply(model.conv1.weight.data, num_bits)

    # print("The size of x is: {}".format(x.size()))
    # print("The size of conv1 is: {}".format(model.conv1.weight.data.size()))
    a = x.size()[0]
    b = model.conv1.weight.data.size()[0]

    memristor_dist1 = torch.normal(1, std, size=[int(a), int(b), 24, 24]).to(device)
    x = F.relu(model.conv1(x) * memristor_dist1)

    with torch.no_grad():
        stats = quant.updateStats(x.clone().view(x.shape[0], -1), stats, 'conv1')

    if act_quant:
        x = quant.FakeQuantOp.apply(x, num_bits, stats['conv1']['ema_min'], stats['conv1']['ema_max'])

    x = F.max_pool2d(x, 2, 2)

    conv2weight = model.conv2.weight.data
    model.conv2.weight.data = quant.FakeQuantOp.apply(model.conv2.weight.data, num_bits)

    # print("The size of x is: {}".format(x.size()))
    # print("The size of conv2 is: {}".format(model.conv1.weight.data.size()))

    a = x.size()[0]
    b = model.conv2.weight.data.size()[0]
    memristor_dist2 = torch.normal(1, std, size=[int(a), int(b), 8, 8]).to(device)
    x = F.relu(model.conv2(x) * memristor_dist2)

    with torch.no_grad():
        stats = quant.updateStats(x.clone().view(x.shape[0], -1), stats, 'conv2')

    if act_quant:
        x = quant.FakeQuantOp.apply(x, num_bits, stats['conv2']['ema_min'], stats['conv2']['ema_max'])

    x = F.max_pool2d(x, 2, 2)
    x = x.view(-1, 4 * 4 * 50)

    fc1weight = model.fc1.weight.data
    model.fc1.weight.data = quant.FakeQuantOp.apply(model.fc1.weight.data, num_bits)
    # print("fc1weight size is: {}".format(fc1weight.size()))

    # print("The size of x is: {}".format(x.size()))
    # print("The size of conv2 is: {}".format(model.fc1.weight.data.size()))

    a = x.size()[0]
    b = model.fc1.weight.data.size()[0]

    # print("The size of a at fc1 is: {}".format(a)) #should be 64
    # print("The size of b at fc1 is: {}".format(b)) # should be 500

    memristor_dist3 = torch.normal(1, std, size=[int(a), int(b)]).to(device)
    x = F.relu(model.fc1(x) * memristor_dist3)
    # print("x after fc1 is: {}".format(x.size()))

    with torch.no_grad():
        stats = quant.updateStats(x.clone().view(x.shape[0], -1), stats, 'fc1')

    if act_quant:
        x = quant.FakeQuantOp.apply(x, num_bits, stats['fc1']['ema_min'], stats['fc1']['ema_max'])

    a = x.size()[0]
    b = model.fc2.weight.data.size()[0]

    memristor_dist4 = torch.normal(1, std, size=[int(a), int(b)]).to(device)
    x = model.fc2(x) * memristor_dist4
    # print("x after fc2 is: {}".format(x.size()))

    with torch.no_grad():
        stats = quant.updateStats(x.clone().view(x.shape[0], -1), stats, 'fc2')

    return F.log_softmax(x, dim=1), conv1weight, conv2weight, fc1weight, stats


def trainQuantAware(args, model, device, train_loader, optimizer, epoch, stats, act_quant=False, num_bits=1, std=0):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output, conv1weight, conv2weight, fc1weight, stats = quantAwareTrainingForward(model, data, stats,
                                                                                       num_bits=num_bits,
                                                                                       act_quant=act_quant, std=std)

        model.conv1.weight.data = conv1weight
        model.conv2.weight.data = conv2weight
        model.fc1.weight.data = fc1weight

        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % args["log_interval"] == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
    return stats


def testQuantAware(args, model, device, test_loader, stats, act_quant, num_bits=1, std=0):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, conv1weight, conv2weight, fc1weight, _ = quantAwareTrainingForward(model, data, stats,
                                                                                       num_bits=num_bits,
                                                                                       act_quant=act_quant, std=std)

            model.conv1.weight.data = conv1weight
            model.conv2.weight.data = conv2weight
            model.fc1.weight.data = fc1weight

            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def mainQuantAware(mnist=True, batch_size=64, test_batch_size=64, epochs=10,
                   lr=0.01, momentum=0.5, seed=1, log_interval=500,
                   save_model=False, no_cuda=False, num_bits=1, std=0):
    use_cuda = not no_cuda and torch.cuda.is_available()

    torch.manual_seed(seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    if mnist:
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=batch_size, shuffle=True, **kwargs)

        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
            batch_size=test_batch_size, shuffle=True, **kwargs)
    else:
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trainset = datasets.CIFAR10(root='./dataCifar', train=True,
                                    download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                   shuffle=True, num_workers=2)

        testset = datasets.CIFAR10(root='./dataCifar', train=False,
                                   download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size,
                                                  shuffle=False, num_workers=2)

    model = NetMem(mnist=mnist).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    args = {}
    args["log_interval"] = log_interval
    stats = {}
    for epoch in range(1, epochs + 1):
        if epoch > 5:
            act_quant = True
        else:
            act_quant = False

        stats = trainQuantAware(args, model, device, train_loader, optimizer, epoch, stats, act_quant,
                                num_bits=num_bits, std=std)
        testQuantAware(args, model, device, test_loader, stats, act_quant, num_bits=num_bits, std=std)

    if (save_model):
        torch.save(model.state_dict(), "mnist_cnn.pt")

    return model, stats

# The following starts the training process

model, old_stats = mainQuantAware(mnist=True, batch_size=64, test_batch_size=64, epochs=3,
                                  lr=0.01, momentum=0.5, seed=1, log_interval=500,
                                  save_model=False, no_cuda=False, num_bits=5, std=0.1)