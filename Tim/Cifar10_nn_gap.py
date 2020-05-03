import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

import PIL

import Pt_nn2

from skimage.transform import resize


classes = [
    'plane', 'car', 'bird', 'cat', 'deer', 'dog',
    'frog', 'horse', 'ship', 'truck']


# Define the neural network that is used to classify images from the CIFAR10
# dataset.
class Cifar10_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_bn = nn.BatchNorm2d(3)

        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(
            in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.conv3_bn = nn.BatchNorm2d(16)

        self.conv4 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv5_bn = nn.BatchNorm2d(32)

        self.conv6 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv7_bn = nn.BatchNorm2d(64)

        self.fc = nn.Linear(64, 10)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.gap = nn.AvgPool2d(kernel_size=8)

        self.conv3_do = nn.Dropout(p=0.1)
        self.conv5_do = nn.Dropout(p=0.25)
        self.conv7_do = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.input_bn(x)
        x = self.conv3_bn(
            self.pool(F.relu(self.conv3(self.conv2(self.conv1(x))))))
        # x = self.conv3_do(x)
        x = self.conv5_bn(self.pool(F.relu(self.conv5(self.conv4(x)))))
        # x = self.conv5_do(x)
        x = F.relu(self.conv6(self.conv5(x)))
        # x_features = x
        x = self.conv7_bn(self.gap(x))
        # x = self.conv7_do(x)
        x = x.view(-1, x.size(1))
        x = self.fc(x)
        return x

    def forward_dbg(self, x):
        x = self.input_bn(x)
        x = self.conv3_bn(
            self.pool(F.relu(self.conv3(self.conv2(self.conv1(x))))))
        # x = self.conv3_do(x)
        x = self.conv5_bn(self.pool(F.relu(self.conv5(self.conv4(x)))))
        # x = self.conv5_do(x)
        x = F.relu(self.conv6(self.conv5(x)))
        x_features = x
        x = self.conv7_bn(self.gap(x))
        # x = self.conv7_do(x)
        x = x.view(-1, x.size(1))
        x = self.fc(x)
        return x, x_features


def load_trainset(
        batch_size=64,
        shuffle=True):

    transform = transforms.Compose(
        [
            transforms.ColorJitter(
                brightness=0.05, contrast=0.05, saturation=0.1, hue=0.05),
            transforms.RandomChoice(
                [
                    transforms.RandomAffine(
                        degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1),
                        resample=PIL.Image.BILINEAR),
                    transforms.RandomResizedCrop((32, 32), scale=(0.8, 1.0))
                ]
            ),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.06), ratio=(1, 1))
        ]
    )

    trainset = torchvision.datasets.CIFAR10(
        root='./data/CIFAR10', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=shuffle, num_workers=2)

    return trainloader


def load_testset(
        batch_size=4,
        shuffle=False):

    transform = transforms.Compose(
        [
            transforms.ToTensor()])

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
            num_epochs=100,
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
            num_epochs=100,
            trainloader=load_trainset(batch_size=50),
            validloader=load_testset(),
            savepath='./data/models/cifar10_cnn.pth')

    # load the data for the heatmap generation
    data = iter(load_testset(shuffle=True)).next()

    model.eval()

    images, labels, act_maps, predictions = Pt_nn.gen_heatmap_gap(model, data)

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
    # x_plot_heatmaps()

    # net = Cifar10_CNN()
    # net.load_state_dict(torch.load('./data/models/cifar10_cnn.pth'))
    # tl = load_trainset()

    # Pt_nn.test_classification_accuracy(net, tl, classes)

    tl = load_trainset()

    Pt_nn.plot_imgs(tl, False)
