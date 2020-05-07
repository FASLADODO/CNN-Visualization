from torchvision.models import vgg13
from torchvision.models import vgg13_bn
from torchvision.models import vgg16
from torchvision.models import vgg16_bn
from torchvision.models import vgg19
from torchvision.models import vgg19_bn

import Pt_imagenet


class VGG13_CNN(Pt_imagenet.GradCAM_CNN):
    def __init__(self):
        super().__init__()

        # load the pretrained VGG16 network
        self.model = vgg13(pretrained=True)

        # dissect the network in order to get access to the feature maps of
        # the convolutional layers
        self.model.features[3].register_backward_hook(self.grad_hook)
        self.model.features[3].register_forward_hook(self.feature_hook)

        self.model.features[8].register_backward_hook(self.grad_hook)
        self.model.features[8].register_forward_hook(self.feature_hook)

        self.model.features[13].register_backward_hook(self.grad_hook)
        self.model.features[13].register_forward_hook(self.feature_hook)

        self.model.features[18].register_backward_hook(self.grad_hook)
        self.model.features[18].register_forward_hook(self.feature_hook)

        self.model.features[23].register_backward_hook(self.grad_hook)
        self.model.features[23].register_forward_hook(self.feature_hook)


class VGG13_BN_CNN(Pt_imagenet.GradCAM_CNN):
    def __init__(self):
        super().__init__()

        # load the pretrained VGG16 network
        self.model = vgg13_bn(pretrained=True)

        # dissect the network in order to get access to the feature maps of
        # the convolutional layers
        self.model.features[5].register_backward_hook(self.grad_hook)
        self.model.features[5].register_forward_hook(self.feature_hook)

        self.model.features[12].register_backward_hook(self.grad_hook)
        self.model.features[12].register_forward_hook(self.feature_hook)

        self.model.features[19].register_backward_hook(self.grad_hook)
        self.model.features[19].register_forward_hook(self.feature_hook)

        self.model.features[26].register_backward_hook(self.grad_hook)
        self.model.features[26].register_forward_hook(self.feature_hook)

        self.model.features[33].register_backward_hook(self.grad_hook)
        self.model.features[33].register_forward_hook(self.feature_hook)


class VGG16_CNN(Pt_imagenet.GradCAM_CNN):
    def __init__(self):
        super().__init__()

        # load the pretrained VGG16 network
        self.model = vgg16(pretrained=True)

        # dissect the network in order to get access to the feature maps of
        # the convolutional layers
        self.model.features[3].register_backward_hook(self.grad_hook)
        self.model.features[3].register_forward_hook(self.feature_hook)

        self.model.features[8].register_backward_hook(self.grad_hook)
        self.model.features[8].register_forward_hook(self.feature_hook)

        self.model.features[15].register_backward_hook(self.grad_hook)
        self.model.features[15].register_forward_hook(self.feature_hook)

        self.model.features[22].register_backward_hook(self.grad_hook)
        self.model.features[22].register_forward_hook(self.feature_hook)

        self.model.features[29].register_backward_hook(self.grad_hook)
        self.model.features[29].register_forward_hook(self.feature_hook)


class VGG16_BN_CNN(Pt_imagenet.GradCAM_CNN):
    def __init__(self):
        super().__init__()

        # load the pretrained VGG16 network
        self.model = vgg16_bn(pretrained=True)

        # dissect the network in order to get access to the feature maps of
        # the convolutional layers
        self.model.features[5].register_backward_hook(self.grad_hook)
        self.model.features[5].register_forward_hook(self.feature_hook)

        self.model.features[12].register_backward_hook(self.grad_hook)
        self.model.features[12].register_forward_hook(self.feature_hook)

        self.model.features[22].register_backward_hook(self.grad_hook)
        self.model.features[22].register_forward_hook(self.feature_hook)

        self.model.features[32].register_backward_hook(self.grad_hook)
        self.model.features[32].register_forward_hook(self.feature_hook)

        self.model.features[42].register_backward_hook(self.grad_hook)
        self.model.features[42].register_forward_hook(self.feature_hook)


class VGG19_CNN(Pt_imagenet.GradCAM_CNN):
    def __init__(self):
        super().__init__()

        # load the pretrained VGG16 network
        self.model = vgg19(pretrained=True)

        # dissect the network in order to get access to the feature maps of
        # the convolutional layers
        self.model.features[3].register_backward_hook(self.grad_hook)
        self.model.features[3].register_forward_hook(self.feature_hook)

        self.model.features[8].register_backward_hook(self.grad_hook)
        self.model.features[8].register_forward_hook(self.feature_hook)

        self.model.features[17].register_backward_hook(self.grad_hook)
        self.model.features[17].register_forward_hook(self.feature_hook)

        self.model.features[26].register_backward_hook(self.grad_hook)
        self.model.features[26].register_forward_hook(self.feature_hook)

        self.model.features[35].register_backward_hook(self.grad_hook)
        self.model.features[35].register_forward_hook(self.feature_hook)


class VGG19_BN_CNN(Pt_imagenet.GradCAM_CNN):
    def __init__(self):
        super().__init__()

        # load the pretrained VGG16 network
        self.model = vgg19_bn(pretrained=True)

        # dissect the network in order to get access to the feature maps of
        # the convolutional layers
        self.model.features[5].register_backward_hook(self.grad_hook)
        self.model.features[5].register_forward_hook(self.feature_hook)

        self.model.features[12].register_backward_hook(self.grad_hook)
        self.model.features[12].register_forward_hook(self.feature_hook)

        self.model.features[25].register_backward_hook(self.grad_hook)
        self.model.features[25].register_forward_hook(self.feature_hook)

        self.model.features[38].register_backward_hook(self.grad_hook)
        self.model.features[38].register_forward_hook(self.feature_hook)

        self.model.features[51].register_backward_hook(self.grad_hook)
        self.model.features[51].register_forward_hook(self.feature_hook)


if __name__ == '__main__':
    # model = VGG13_CNN()
    # model = VGG13_BN_CNN()
    # model = VGG16_CNN()
    # model = VGG16_BN_CNN()
    model = VGG19_CNN()
    # model = VGG19_BN_CNN()
    Pt_imagenet.plot_imagenet_heatmaps(model, [0, 1, 2], [0, 2, 4])
