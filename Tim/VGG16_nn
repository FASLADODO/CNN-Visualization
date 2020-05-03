import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

from torchvision import datasets
from torchvision.models import vgg16

import Pt_nn

from skimage.transform import resize


# Define the neural network that is used to classify images from the CIFAR10
# dataset.
class VGG16_CNN(nn.Module):
    def __init__(self):
        super().__init__()

        # load the pretrained VGG16 network
        self.vgg16 = vgg16(pretrained=True)

        # dissect the network in order to get access to the feature maps of
        # the convolutional layers
        self.features = self.vgg16.features[:30]

        # We have to add the missing max pooling operation again...
        self.max_pool = self.vgg16.features[30]

        # Extract the remaining layers of the VGG16 network
        self.avg_pool = self.vgg16.avgpool
        self.classifier = self.vgg16.classifier

        # Create class variable to store the gradients
        self.gradients = None

    def grad_hook(self, gradients):
        self.gradients = gradients

    def get_feature_gradients(self, feature_layer=None):
        if feature_layer is None:
            return self.gradients

    def get_feature_maps(self, x, feature_layer=None):
        if feature_layer is None:
            x = self.features(x)
            return x

    def forward(self, x):
        x = self.get_feature_maps(x)

        x.register_hook(self.grad_hook)

        x = self.max_pool(x)
        x = self.avg_pool(x)
        x = x.view((x.shape[0], -1))
        x = self.classifier(x)
        return x


def load_trainset(batch_size=4):
    # Use the standard normalization for ImageNet
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])

    trainset = torchvision.datasets.ImageNet(
        root='./Tim/data/ImageNet',
        train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    return trainloader


def load_testset(batch_size=1, normalized=True, shuffle=True):
    # Use the standard normalization for ImageNet
    if normalized:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.ToTensor()

    testset = datasets.ImageFolder(
        root='./Tim/data/ImageNet/', transform=transform)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=shuffle, num_workers=2)

    return testloader


def x_gen_heatmap():
    # load the model
    model = VGG16_CNN()

    # load the data for the heatmap generation
    data = iter(load_testset(shuffle=False)).next()

    images, labels, act_maps, pred_indcs = Pt_nn.gen_heatmap_grad(
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
    tl = load_testset(batch_size=1, shuffle=False)
    input_image, label = iter(tl).next()

    tl = load_testset(batch_size=1, normalized=False, shuffle=False)
    image, _ = iter(tl).next()
    image = torch.squeeze(image)

    model = VGG16_CNN()

    Pt_nn.gen_heatmap_grad(model, input_image, image)
