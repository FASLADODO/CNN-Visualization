from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np

from UtilFunctions import get_test_params, save_cam_imgs


class ScoreCAM():
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

        # Create class variable to store the feature maps.
        self.feature_maps = []

        self.store_feature_maps = True

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
        if self.store_feature_maps:
            self.feature_maps.append(output)

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
        Register a forward hook to the given layer.

        Args:
            layer (Pytorch module): Layer of the current CNN model.

        returns:
            None
        """
        layer.register_forward_hook(self.feature_hook)

    def get_CAM(self, input_image, target_class=None, layer_idx=0):
        """
        Generate a class activation map for one layer of the CNN. The class
        activation map is computed by a weighted sum of the feature maps,
        where the weights are the softmax of the scores that are achieved by
        the CNN if the input image is multiplied by the feature map. CAMs
        can only be computed for the layers which have a hook for retrieving
        the information attached to it. These layers are indexed by the
        layer_idx variable. The layer_idx starts with zero for the last hooked
        layer and counts up towards the first layer that has a hook attached
        to it.

        Args:
            input_image (Pytorch tensor): Image of dimension (B,D,W,H) with
                B=1. CAM is generated with respect to this image.
            target_class (int): Classlabel. The CAM is generated with respect
                to this class. If no class is given, the target class is
                chosen to be the predicted class of the CNN.
            layer_idx (int): Index of the layer for which the CAM should be
                generated.

        returns:
            cam (Numpy ndArray): Grayscale image in range [0,255]
        """
        # Do the forward pass through the model.
        model_prediction = self.model(input_image)

        # Don't store feature maps when generating the CAM
        self.store_feature_maps = False

        if target_class is None:
            # Set the target class to the predicted class.
            target_class = torch.argmax(model_prediction)

        # Get the feature maps at the layer of the target index.
        features = torch.tensor(
            self.get_feature_maps(layer_idx).data.numpy()[0])

        w = np.zeros(features.shape[0])

        # Calculate the weighted sum of the feature maps.
        for i in range(features.shape[0]):
            act_map = torch.unsqueeze(torch.unsqueeze(features[i, :, :], 0), 0)
            # Upsampling to input size
            act_map = F.interpolate(
                act_map, size=(224, 224), mode='bilinear', align_corners=False)
            if act_map.max() == act_map.min():
                continue
            # Scale between 0-1
            norm_act_map = (act_map - act_map.min()) /\
                (act_map.max() - act_map.min())

            # Get the target score
            with torch.no_grad():
                w[i] = self.model(input_image*norm_act_map)[0, target_class]

        # normalize weights
        norm_weights = F.softmax(torch.tensor(w), dim=0)

        # Create empty numpy array to store the class activation map.
        cam = torch.zeros(
            features.shape[1], features.shape[2],
            dtype=torch.float32)
        # Compute CAM as a weighted sum of the feature maps
        for i in range(features.shape[0]):
            feat_map = features[i, :, :].data.numpy()
            if np.max(feat_map) == np.min(feat_map):
                continue
            norm_feat_map = (feat_map - np.min(feat_map)) /\
                (np.max(feat_map) - np.min(feat_map))
            cam += norm_weights[i] * norm_feat_map

        cam = cam.numpy()
        # Only take positive ativations into account.
        cam = np.maximum(0, cam)  # ReLU
        # Normalize the CAM range [0,1]
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
        # Convert to int in range [0, 255]
        cam = np.uint8(255 * cam)
        return cam


if __name__ == '__main__':
    # Get params
    img_indx = 2  # Snake
    layer_indcs = [0, 1, 2, 3, 4]
    file_name = 'scoreCAM/Score-CAM_img' + str(img_indx)
    # Get the input parameters.
    (original_image, input_img, target_class, model) =\
        get_test_params(img_indx)
    # Score cam
    score_cam = ScoreCAM(model)
    # Register the hooks.
    score_cam.register_hook(score_cam.model.features[5])
    score_cam.register_hook(score_cam.model.features[12])
    score_cam.register_hook(score_cam.model.features[22])
    score_cam.register_hook(score_cam.model.features[32])
    score_cam.register_hook(score_cam.model.features[42])

    # Save mask
    for i in layer_indcs:
        cam = score_cam.get_CAM(
            input_img, layer_idx=i, target_class=target_class)
        tmp_file_name = file_name + '_f' + str(i)
        # Save mask
        save_cam_imgs(
            original_image, cam, tmp_file_name)
        print('Generated CAM: ' + tmp_file_name)
    print('Score cam completed')
