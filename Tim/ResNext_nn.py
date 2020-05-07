from torchvision.models import resnext50_32x4d
from torchvision.models import resnext101_32x8d

import Pt_imagenet


# Define the neural network that is used to classify images from the CIFAR10
# dataset.
class ResNext101_32x8d_CNN(Pt_imagenet.GradCAM_CNN):
    def __init__(self):
        super().__init__()

        # load the pretrained VGG16 network
        self.model = resnext101_32x8d(pretrained=True)

        # dissect the network in order to get access to the feature maps of
        # the convolutional layers
        self.model.layer1.register_backward_hook(self.grad_hook)
        self.model.layer1.register_forward_hook(self.feature_hook)

        self.model.layer2.register_backward_hook(self.grad_hook)
        self.model.layer2.register_forward_hook(self.feature_hook)

        self.model.layer3.register_backward_hook(self.grad_hook)
        self.model.layer3.register_forward_hook(self.feature_hook)

        self.model.layer4.register_backward_hook(self.grad_hook)
        self.model.layer4.register_forward_hook(self.feature_hook)


# Define the neural network that is used to classify images from the CIFAR10
# dataset.
class ResNext50_32x4d_CNN(Pt_imagenet.GradCAM_CNN):
    def __init__(self):
        super().__init__()

        # load the pretrained VGG16 network
        self.model = resnext50_32x4d(pretrained=True)

        # dissect the network in order to get access to the feature maps of
        # the convolutional layers
        self.model.layer1.register_backward_hook(self.grad_hook)
        self.model.layer1.register_forward_hook(self.feature_hook)

        self.model.layer2.register_backward_hook(self.grad_hook)
        self.model.layer2.register_forward_hook(self.feature_hook)

        self.model.layer3.register_backward_hook(self.grad_hook)
        self.model.layer3.register_forward_hook(self.feature_hook)

        self.model.layer4.register_backward_hook(self.grad_hook)
        self.model.layer4.register_forward_hook(self.feature_hook)


if __name__ == '__main__':
    model = ResNext50_32x4d_CNN()
    # model = ResNext101_32x8d_CNN()
    Pt_imagenet.plot_imagenet_heatmaps(model, [24, 25], [0, 1, 2, 3])
