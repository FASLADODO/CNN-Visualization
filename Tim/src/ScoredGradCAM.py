import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from UtilFunctions import get_test_params, save_cam_imgs


class GradCAM():
    '''
    Helper class to generate class activation maps for the given model.
    '''
    def __init__(self, model):
        """
        Set the model and initalize a list for the gradient and feature maps.

        Args:
            model (Pytorch model): CNN model. Class activation maps are
                generated for this model.

        returns:
            None
        """
        self.model = model
        self.model.eval()
        self.model.zero_grad()

        # Create class variable to store the gradients.
        self.gradients = []

        # Create class variable to store the feature maps.
        self.feature_maps = []

    def grad_hook(self, layer, input, output):
        """
        Function is intended to be hooked to a layer of a CNN. Stores the
        gradient input of the layer it is hooked to in a local gradients list.

        Args:
            layer (Pytorch module): Reference to the layer that the function
                is hooked to.
            input (Pytorch tensor): Gradients that are passed to the next
                layer.
            output (Pytorch tensor): Gradients that are passed into the layer.

        returns:
            None
        """
        self.gradients.append(output[0])

    def feature_hook(self, layer, input, output):
        """
        Function is intended to be hooked to a layer of a CNN. Stores the
        feature input of the layer it is hooked to in a local features list.

        Args:
            layer (Pytorch module): Reference to the layer that the function
                is hooked to.
            input (Pytorch tensor): Feature maps that are passed into the
                layer.
            output (Pytorch tensor): Feature maps that are passed to the next
                layer.

        returns:
            None
        """
        self.feature_maps.append(output)

    def get_gradients(self, layer_idx=0):
        """
        Returns one of the stored gradient maps. The layer_idx starts with zero
        for the last hooked layer and counts up towards the first layer that
        has a hook attached to it.

        Args:
            layer_idx (int): Identifier for the layer.

        returns:
            self.gradients[layer_idx] (Pytorch tensor): Gradient map for the
                specified layer.
        """
        return self.gradients[layer_idx]

    def get_feature_maps(self, layer_idx=0):
        """
        Returns one of the stored feature maps. The layer_idx starts with zero
        for the last hooked layer and counts up towards the first layer that
        has a hook attached to it.

        Args:
            layer_idx (int): Identifier for the layer.

        returns:
            self.feature_maps[layer_idx] (Pytorch tensor): Feature map for the
                specified layer.
        """
        layer_idx = len(self.feature_maps) - layer_idx - 1
        return self.feature_maps[layer_idx]

    def register_hook(self, layer):
        """
        Register a forward and a backward hook to the given layer.

        Args:
            layer (Pytorch module): Layer of the curren CNN model.

        returns:
            None
        """
        layer.register_backward_hook(self.grad_hook)
        layer.register_forward_hook(self.feature_hook)

    def generate_CAM(self, input_image, target_class=None):
        """
        Generate and store feature activation maps and gradient maps with
        respect to the given input image and target class. The CAM is
        generated by get_CAM by computing a weighted sum of the feature
        maps in one layer of the CNN. The weights are the global average
        pooled gradient maps of this particular layer.

        Args:
            input_image (Pytorch tensor): Image of dimension (B,D,W,H) with
                B=1. CAM is generated with respect to this image.
            target_class (int): Classlabel. The CAM is generated with respect
                to this class. If no class is given, the target class is
                chosen to be the predicted class of the CNN.

        returns:
            None
        """
        self.gradients = []
        self.feature_maps = []

        # Do the forward pass through the model.
        model_prediction = self.model(input_image)

        if target_class is None:
            # Set the target class to the predicted class.
            target_class = torch.argmax(model_prediction)

        # Calculate gradients with respect to the target class.
        self.model.zero_grad()
        model_prediction[0, target_class].backward()
        return target_class

    def get_CAM(self, layer_idx=0):
        """
        Generate a class activation map for one layer of the CNN. The class
        activation map is computed by a weighted sum of the feature maps,
        where the weights are the global average pooled gradient maps. CAMs
        can only be computed for the layers for which the gradient maps and
        feature maps where stored in the generate_CAM, i.e. for the layers
        which have a hook for retrieving the information attached to it. These
        layers are indexed by the layer_idx variable. The layer_idx starts
        with zero for the last hooked layer and counts up towards the first
        layer that has a hook attached to it.

        Args:
            target_idx (int): Identifier for the layer.

        returns:
            cam (Numpy ndArray): Grayscale image in range [0,255]
        """
        # Get the gradients at the layer of the target index.
        gradients = self.get_gradients(layer_idx).data.numpy()
        # Get the feature maps at the layer of the target index.
        features = self.get_feature_maps(layer_idx).data.numpy()
        # Calculate the weights.
        weights = np.mean(gradients, axis=(2, 3))
        weights = weights[0, :]
        # Create a numpy array to store the class activation map.
        cam = np.ones(features.shape[2:], dtype=np.float32)
        # Calculate the weighted sum of the feature maps.
        for i, w in enumerate(weights):
            cam += w * features[0, i, :, :]
        # Only take positive ativations into account.
        cam = np.maximum(0, cam)  # ReLU
        # Normalize the CAM range [0,1]
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
        # Convert to int in range [0, 255]
        # cam = np.uint8(255 * cam)

        return cam

    def get_score_grad_cam(self, input_img, num_cams, target_class=None):
        cam_scaled = np.zeros(
            (num_cams, input_img.shape[2], input_img.shape[3]))
        scores = torch.zeros(5)

        for i in range(num_cams):
            cam = grad_cam.get_CAM(i)
            # Resize and store mask
            cam_scaled[i] = np.array(Image.fromarray(cam/np.max(cam)).resize(
                (input_img.shape[3], input_img.shape[2]),
                Image.ANTIALIAS))

            weighted_input = torch.ones_like(input_img)
            weighted_input[0] = torch.Tensor(cam_scaled[i]) * input_img[0]
            pred = model(weighted_input)
            scores[i] = torch.softmax(pred, 1)[0, target_class] / np.sum(
                cam_scaled[i], (0, 1)) * cam_scaled.shape[1] *\
                cam_scaled.shape[2]

        scores = torch.softmax(scores, dim=0)
        print(scores)
        cam_avg = np.zeros((input_img.shape[2], input_img.shape[3]))

        for i in range(scores.shape[0]):
            cam_avg += scores[i].detach().numpy() * cam_scaled[i]

        return cam_avg


if __name__ == '__main__':
    # Set the parameters for the input data
    img_indx = 12  # Snake
    layer_indcs = [0, 1, 2, 3, 4]
    file_name = 'scoredGradCAM/Scored-Grad-CAM_img' + str(img_indx)

    # Get the input parameters.
    (original_image, input_img, target_class, model) =\
        get_test_params(img_indx)

    # Initialize Grad-CAM object.
    grad_cam = GradCAM(model)

    # Register the hooks.
    grad_cam.register_hook(grad_cam.model.features[5])
    grad_cam.register_hook(grad_cam.model.features[12])
    grad_cam.register_hook(grad_cam.model.features[22])
    grad_cam.register_hook(grad_cam.model.features[32])
    grad_cam.register_hook(grad_cam.model.features[42])
    # Generate cam mask
    target_class = grad_cam.generate_CAM(input_img, None)

    # cam_scaled = np.zeros(
    #    (5, input_img.shape[2], input_img.shape[3]))

    # scores = torch.zeros(5)

    #for i in layer_indcs:
    #    cam = grad_cam.get_CAM(i)
    #    tmp_file_name = file_name + '_f' + str(i)
        # Resize and store mask
    #    cam_scaled[i] = np.array(Image.fromarray(cam/np.max(cam)).resize(
    #        (input_img.shape[3], input_img.shape[2]),
    #        Image.ANTIALIAS))
    #    save_cam_imgs(
    #        original_image.data.numpy(), cam, tmp_file_name)

    #    weighted_input = torch.ones_like(input_img)
    #    weighted_input[0] = torch.Tensor(cam_scaled[i]) * input_img[0]
    #    # plt.imshow(np.transpose(weighted_input[0].data.numpy(), [1, 2, 0]))
    #    # plt.show()
    #    pred = model(weighted_input)
    #    scores[i] = torch.softmax(pred, 1)[0, target_class] / np.sum(cam_scaled[i], (0, 1)) *cam_scaled.shape[1] * cam_scaled.shape[2]
    #    print(scores[i])
    #    print('Generated CAM: ' + tmp_file_name)
    # scores = torch.softmax(scores, dim=0)
    # print(scores)

    # cam_avg = np.zeros((input_img.shape[2], input_img.shape[3]))
    # for i in range(scores.shape[0]):
    #    cam_avg += scores[i].detach().numpy() * cam_scaled[i]

    # plt.matshow(cam_avg)
    # plt.show()

    cam_avg = grad_cam.get_score_grad_cam(
        input_img, 5, target_class)

    tmp_file_name = file_name + '_avg'
    save_cam_imgs(
        original_image.data.numpy(), cam_avg, tmp_file_name)

    print('Scored Grad cam completed')
