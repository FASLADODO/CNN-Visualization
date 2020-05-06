import torch
import torch.nn as nn
import torchvision.transforms as transforms

from torchvision import datasets
from torchvision.models import vgg16

import matplotlib.pyplot as plt

import ImageNet_classlabels as IN_labels

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
        self.vgg16.features[3].register_backward_hook(self.grad_hook)
        self.vgg16.features[3].register_forward_hook(self.feature_hook)

        self.vgg16.features[8].register_backward_hook(self.grad_hook)
        self.vgg16.features[8].register_forward_hook(self.feature_hook)

        self.vgg16.features[15].register_backward_hook(self.grad_hook)
        self.vgg16.features[15].register_forward_hook(self.feature_hook)

        self.vgg16.features[22].register_backward_hook(self.grad_hook)
        self.vgg16.features[22].register_forward_hook(self.feature_hook)

        self.vgg16.features[29].register_backward_hook(self.grad_hook)
        self.vgg16.features[29].register_forward_hook(self.feature_hook)

        # Create class variable to store the gradients
        self.gradients = []

        self.feature_maps = []

    def grad_hook(self, layer, input, output):
        self.gradients.append(input[0])

    def feature_hook(self, layer, input, output):
        self.feature_maps.append(output)

    def get_feature_gradients(self, feature_layer=None):
        grad_indx = 0
        if feature_layer is not None:
            grad_indx = feature_layer
        return self.gradients[grad_indx]

    def get_feature_maps(self, x, feature_layer=None):
        feat_map_indx = len(self.feature_maps) - 1
        if feature_layer is not None:
            feat_map_indx = len(self.feature_maps) - feature_layer - 1
        return self.feature_maps[feat_map_indx]

    def get_num_featuremaps(self):
        return len(self.feature_maps)

    def reset(self):
        self.zero_grad()
        self.gradients = []
        self.feature_maps = []

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


def x_gen_heatmap(model, imgIndx=0, feature_layers=[0]):
    tl_norm = iter(load_testset(batch_size=1, normalized=True, shuffle=False))
    tl = iter(load_testset(batch_size=1, normalized=False, shuffle=False))

    # load the data for the heatmap generation
    for i in range(imgIndx+1):
        input_image, label = tl_norm.next()
        image, label = tl.next()

    heatmaps, pred_indx, pred_certainty = Pt_nn.gen_heatmap_grad(
        model, input_image, torch.squeeze(image),
        feature_layers=feature_layers)

    return heatmaps, pred_indx, pred_certainty


def x_plot_heatmaps(image_indcs, feature_indcs):
    nrows = len(image_indcs)
    ncols = len(feature_indcs)

    height = 6
    width = 8

    # load the model
    model = VGG16_CNN()

    fig, axs = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=[height, width])

    for ax in axs.flat:
        ax.axis('off')

    for img_indx in range(len(image_indcs)):
        heatmap, pred_indx, pred_certainty = x_gen_heatmap(
            model, imgIndx=image_indcs[img_indx], feature_layers=feature_indcs)
        for feature_indx in range(len(feature_indcs)):
            x = img_indx
            y = feature_indx

            axs[x, y].imshow(heatmap[feature_indx])
            axs[x, y].set_title(
                "Prediction: " + IN_labels.classlabels[pred_indx.item()])

    plt.show()


if __name__ == '__main__':
    # tl = load_testset(batch_size=1, shuffle=False)
    # input_image, label = iter(tl).next()

    # tl = load_testset(batch_size=1, normalized=False, shuffle=False)
    # image, _ = iter(tl).next()
    # image = torch.squeeze(image)

    # model = VGG16_CNN()

    # Pt_nn.gen_heatmap_grad(model, input_image, image)

    x_plot_heatmaps([24, 25], [0, 2, 4])
