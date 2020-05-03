import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

import Pt_nn

from skimage.transform import resize


# Define the neural network that is used to classify images from the CIFAR10
# dataset.
class MNIST_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=16,
            kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32,
            kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=32,
            kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(
            in_channels=32, out_channels=10,
            kernel_size=3, stride=1, padding=1)

        self.gap = nn.AvgPool2d(kernel_size=7)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.gap(x)
        x = x.view(-1, x.size(1))
        return x

    def forward_dbg(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x_features = F.relu(self.conv4(x))
        pred = self.gap(x_features)
        x = x.view(-1, x.size(1))
        return pred.detach(), x_features.detach()


def load_trainset(batch_size=4):
    transform = transforms.Compose(
        [
            transforms.ToTensor()])

    trainset = torchvision.datasets.MNIST(
        root='./data/MNIST', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    return trainloader


def load_testset(batch_size=4, shuffle=False):
    transform = transforms.Compose(
        [
            transforms.ToTensor()])

    testset = torchvision.datasets.MNIST(
        root='./data/MNIST', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=shuffle, num_workers=2)

    return testloader


def x_train_MNIST_nn():
    # generate and train the CNN
    model = MNIST_CNN()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    model = Pt_nn.train_network(
            model=model,
            optimizer=optimizer,
            loss_func=nn.CrossEntropyLoss(),
            num_epochs=10,
            trainloader=load_trainset(batch_size=50),
            validloader=load_testset(),
            savepath='./data/models/mnist_cnn_gap.pth')


def x_gen_heatmap():
    # load the model
    model = MNIST_CNN()
    try:
        model.load_state_dict(torch.load("./data/models/mnist_cnn_gap.pth"))
    except Exception:
        model = MNIST_CNN()
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        model = Pt_nn.train_network(
            model=model,
            optimizer=optimizer,
            loss_func=nn.CrossEntropyLoss(),
            num_epochs=10,
            trainloader=load_trainset(batch_size=50),
            validloader=load_testset(batch_size=50),
            savepath='./data/models/mnist_cnn_gap.pth')

    # load the data for the heatmap generation
    data = iter(load_testset(shuffle=True)).next()

    images, labels, act_maps, pred_indcs = Pt_nn.gen_heatmap_pure_gap(
        model, data)
    return images, labels, act_maps, pred_indcs.view(-1, pred_indcs.size(1))


def x_plot_heatmaps():
    images, labels, act_maps, pred_indcs = x_gen_heatmap()

    scaled_act_maps = torch.zeros([
        images.size()[0],       # batch dimension
        act_maps.size()[1],     # feature map
        images.size()[2],       # x-dim
        images.size()[3]])      # y-dim

    # scale the activation maps to the size of the original images
    # and normalize the dynamic range of each image to a range of [0, 1]
    for img_indx in range(act_maps.size()[0]):
        for feature_indx in range(act_maps.size()[1]):
            scaled_act_maps[img_indx, feature_indx, :, :] =\
                torch.tensor(resize(
                    act_maps[img_indx, feature_indx, :, :].detach().numpy(),
                    images.size()[2:]))

    Pt_nn.plot_heatmaps(images, labels, scaled_act_maps, pred_indcs, 5)


if __name__ == '__main__':
    x_plot_heatmaps()
