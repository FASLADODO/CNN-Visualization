from torchvision.models import resnet18
from torchvision.models import resnet34
from torchvision.models import resnet50
from torchvision.models import resnet101
from torchvision.models import resnet152

import Pt_imagenet


# Define the neural network that is used to classify images from the CIFAR10
# dataset.
class ResNet18_CNN(Pt_imagenet.GradCAM_CNN):
    def __init__(self):
        super().__init__()

        # load the pretrained VGG16 network
        self.model = resnet18(pretrained=True)

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


class ResNet34_CNN(Pt_imagenet.GradCAM_CNN):
    def __init__(self):
        super().__init__()

        # load the pretrained VGG16 network
        self.model = resnet34(pretrained=True)

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


class ResNet50_CNN(Pt_imagenet.GradCAM_CNN):
    def __init__(self):
        super().__init__()

        # load the pretrained VGG16 network
        self.model = resnet50(pretrained=True)

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


class ResNet101_CNN(Pt_imagenet.GradCAM_CNN):
    def __init__(self):
        super().__init__()

        # load the pretrained VGG16 network
        self.model = resnet101(pretrained=True)

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


class ResNet152_CNN(Pt_imagenet.GradCAM_CNN):
    def __init__(self):
        super().__init__()

        # load the pretrained VGG16 network
        self.model = resnet152(pretrained=True)

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
    # model = ResNet18_CNN()
    # model = ResNet34_CNN()
    # model = ResNet50_CNN()
    # model = ResNet101_CNN()
    model = ResNet152_CNN()
    Pt_imagenet.plot_imagenet_heatmaps(model, [24, 25], [0, 1, 2, 3])
