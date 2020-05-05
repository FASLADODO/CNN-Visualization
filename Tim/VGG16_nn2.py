import torch
import torch.nn as nn
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
        self.vgg16.features[29].register_backward_hook(self.grad_hook)
        self.vgg16.features[29].register_forward_hook(self.feature_hook)

        # Create class variable to store the gradients
        self.gradients = None

        self.feature_maps = None

    def grad_hook(self, layer, input, output):
        self.gradients = input[0]

    def feature_hook(self, layer, input, output):
        self.feature_maps = output

    def get_feature_gradients(self, feature_layer=None):
        if feature_layer is None:
            return self.gradients

    def get_feature_maps(self, x, feature_layer=None):
        if feature_layer is None:
            x = self.feature_maps
            return x

    def forward(self, x):
        x = self.vgg16(x)
        return x


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
        root='./data/ImageNet/', transform=transform)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=shuffle, num_workers=2)

    return testloader


def x_gen_heatmap(imgIndx=0):
    # load the model
    model = VGG16_CNN()

    # load the data for the heatmap generation
    for i in range(imgIndx+1):
        input_image, label = iter(load_testset(shuffle=False)).next()

    images, labels, act_maps, pred_indcs = Pt_nn.gen_heatmap_grad(
        model, input_image)
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
