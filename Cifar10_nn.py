import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

import Pt_nn

from skimage.transform import resize


classes = (
    'plane', 'car', 'bird', 'cat', 'deer', 'dog',
    'frog', 'horse', 'ship', 'truck')


# Define the neural network that is used to classify images from the CIFAR10
# dataset.
class Cifar10_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(
            in_channels=8, out_channels=16, kernel_size=3, padding=1)

        self.conv3 = nn.Conv2d(
            in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(
            in_channels=16, out_channels=24, kernel_size=3, padding=1)

        self.conv5 = nn.Conv2d(
            in_channels=24, out_channels=24, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(
            in_channels=24, out_channels=24, kernel_size=3, padding=1)

        self.fc = nn.Linear(24, 10)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.gap = nn.AvgPool2d(kernel_size=8)

    def forward(self, x):
        x = self.pool(F.relu(self.conv2(self.conv1(x))))
        x = self.pool(F.relu(self.conv4(self.conv3(x))))
        x = self.gap(F.relu(self.conv6(self.conv5(x))))
        x = x.view(-1, x.size(1))
        x = self.fc(x)
        return x

    def forward_dbg(self, x):
        x_conv2 = self.pool(F.relu(self.conv2(self.conv1(x))))
        x_conv4 = self.pool(F.relu(self.conv4(self.conv3(x_conv2))))
        x_conv6 = F.relu(self.conv6(self.conv5(x_conv4)))
        x_conv6_vec = self.gap(x_conv6)
        x_conv6_vec = x_conv6_vec.view(-1, x_conv6_vec.size(1))
        pred = self.fc(x_conv6_vec)
        return pred.detach(), x_conv6.detach()


def load_trainset(
        batch_size=4,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        shuffle=True):

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])

    trainset = torchvision.datasets.CIFAR10(
        root='./data/CIFAR10', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=shuffle, num_workers=2)

    return trainloader


def load_testset(
        batch_size=4,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        shuffle=False):

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])

    testset = torchvision.datasets.CIFAR10(
        root='./data/CIFAR10', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=shuffle, num_workers=2)

    return testloader


def x_train_Cifar10_nn():
    # generate and train the CNN
    model = Cifar10_CNN()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    model = Pt_nn.train_network(
            model=model,
            optimizer=optimizer,
            loss_func=nn.CrossEntropyLoss(),
            num_epochs=10,
            trainloader=load_trainset(batch_size=50),
            validloader=load_testset(),
            savepath='./data/models/cifar10_cnn.pth')


def x_gen_heatmap():
    # load the model
    model = Cifar10_CNN()
    try:
        model.load_state_dict(torch.load("./data/models/cifar10_cnn.pth"))
    except Exception:
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        model = Pt_nn.train_network(
            model=model,
            optimizer=optimizer,
            loss_func=nn.CrossEntropyLoss(),
            num_epochs=10,
            trainloader=load_trainset(batch_size=50),
            validloader=load_testset(),
            savepath='./data/models/model.pth')

    # load the data for the heatmap generation
    data = iter(load_testset(shuffle=True)).next()

    images, labels, act_maps, predictions = Pt_nn.gen_heatmap_gap(model, data)

    images = images * 0.5 + 0.5

    return images, labels, act_maps, predictions


def x_plot_heatmaps():
    images, labels, act_maps, predictions = x_gen_heatmap()

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

    Pt_nn.plot_heatmaps(images, labels, scaled_act_maps, predictions, 5)


if __name__ == '__main__':
    x_plot_heatmaps()
