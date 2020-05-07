import torch
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import numpy as np

import matplotlib.pyplot as plt

import Pt_nn

import ImageNet_classlabels as IN_labels


# Define the neural network that is used to classify images from the CIFAR10
# dataset.
class GradCAM_CNN(nn.Module):
    def __init__(self):
        super().__init__()

        # variable to store the pretrained network
        self.model = None

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
        x = self.model(x)
        return x


# helper function to show an image
# (used in the `plot_classes_preds` function below)
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

    plt.show()


def plot_imgs(loader, one_channel=True):
    # get some random training images
    dataiter = iter(loader)
    images, labels = dataiter.next()

    # create grid of images
    img_grid = torchvision.utils.make_grid(images)

    # show images
    matplotlib_imshow(img_grid, one_channel=one_channel)


def load_imagenet_testset(batch_size=1, normalized=True, shuffle=True):
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


def gen_imagenet_heatmap(model, imgIndx=0, feature_layers=[0]):
    tl_norm = iter(load_imagenet_testset(
        batch_size=1, normalized=True, shuffle=False))
    tl = iter(load_imagenet_testset(
        batch_size=1, normalized=False, shuffle=False))

    # load the data for the heatmap generation
    for i in range(imgIndx+1):
        input_image, label = tl_norm.next()
        image, label = tl.next()

    heatmaps, pred_indx, pred_certainty = Pt_nn.gen_heatmap_grad(
        model, input_image, torch.squeeze(image),
        feature_layers=feature_layers)

    return heatmaps, pred_indx, pred_certainty


def plot_imagenet_heatmaps(model, image_indcs, feature_indcs):
    nrows = len(image_indcs)
    ncols = len(feature_indcs)

    height = 6
    width = 8

    fig, axs = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=[height, width])

    for ax in axs.flat:
        ax.axis('off')

    for img_indx in range(len(image_indcs)):
        heatmap, pred_indx, pred_certainty = gen_imagenet_heatmap(
            model, imgIndx=image_indcs[img_indx], feature_layers=feature_indcs)
        for feature_indx in range(len(feature_indcs)):
            x = img_indx
            y = feature_indx

            axs[x, y].imshow(heatmap[feature_indx])
            axs[x, y].set_title(
                "Prediction: " + IN_labels.classlabels[pred_indx.item()])

    plt.show()
