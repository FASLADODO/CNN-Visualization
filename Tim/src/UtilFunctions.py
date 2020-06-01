import os
import copy
import numpy as np
from PIL import Image
import matplotlib.cm as plt_cmap

import torch
from torchvision import models
from torchvision import transforms
from torchvision import datasets


def img_to_colormap(orig_img, activation_map, colormap_type):
    '''
    Converts an RGB image (D,W,H) to a grayscale image (1,W,H).

    Params:
        orig_img (numpy array): RGB image with shape (D,W,H)
        activation_map (numpy array): Grayscale image with shape (W,H)
        colormap_type (string): String identifier for the type of color map

    Returns:
        colormap (PIL image): RGB image of the pure colormap generated from
            the activation map.
        colormap_on_image (PIL image): RGB image of the colormap that is
            overlayed over the orig_img.
    '''
    # Convert the source image to PIL image.
    orig_img = np.transpose(orig_img, (1, 2, 0))
    orig_img = Image.fromarray((orig_img*255).astype(np.uint8))

    # Get a matplotlib colormap.
    plt_colormap = plt_cmap.get_cmap(colormap_type)
    colormap = plt_colormap(activation_map)
    # Copy the heatmap since we want to return the unmodified heatmap and the
    # overlayed map.
    colormap = copy.copy(colormap)
    # Make the heatmap transparent for overlaying it onto the original image.
    colormap[:, :, 3] = 0.4
    colormap = Image.fromarray((colormap*255).astype(np.uint8))
    colormap = colormap.resize(orig_img.size)

    # Overlay the heatmap and the image.
    colormap_on_image = Image.new("RGBA", orig_img.size)
    colormap_on_image = Image.alpha_composite(
        colormap_on_image, orig_img.convert('RGBA'))
    colormap_on_image = Image.alpha_composite(colormap_on_image, colormap)
    return colormap, colormap_on_image


def img_to_grayscale(img):
    '''
    Converts an RGB image (D,W,H) to a grayscale image (1,W,H).

    Params:
        img_as_array (numpy array): RGB image with shape (D,W,H)

    Returns:
        grayscale_img (numpy array): Grayscale image with shape (1,W,H)
    '''
    grayscale_img = np.sum(np.abs(img), axis=0) / 3
    grayscale_img = np.expand_dims(grayscale_img, axis=0)
    return grayscale_img


def save_cam_imgs(orig_img, cam_img, file_name):
    '''
    Save the class activation map a grayscale image, with colormap applied and
    overlayed over the original image. The naming convention is as follows:
        Grayscale CAM is saved as:
            - file_name + '_CAM_Grayscale.png'
        Colored CAM is saved as:
            - file_name + '_CAM_Heatmap.png'
        Colored CAM that is overlayed over the original image is saved as:
            - file_name + '_CAM_on_Image.png'

    Params:
        orig_img (numpy array): RGB image with shape (D,W,H)
        cam_img (numpy array): Grayscale image with shape (W,H)
        file_name (string): Prefix of the filename that the images are saved
            to.

    Returns:
        None
    '''
    cam_img = np.uint8(cam_img / np.max(cam_img) * 255)
    cam_img = np.uint8(Image.fromarray(cam_img).resize(
        (orig_img.shape[2], orig_img.shape[1]), Image.ANTIALIAS))/255

    if not os.path.exists('./results'):
        os.makedirs('./results')

    # Convert grayscale activation map to colored heatmap.
    heatmap, heatmap_on_image = img_to_colormap(orig_img, cam_img, 'jet')
    # Save colored heatmap
    path_to_file = os.path.join('./results', file_name + '_CAM_Heatmap.png')
    save_image(heatmap, path_to_file)
    # Save heatmap on iamge
    path_to_file = os.path.join('./results', file_name + '_CAM_on_Image.png')
    save_image(heatmap_on_image, path_to_file)
    # SAve grayscale heatmap
    path_to_file = os.path.join('./results', file_name + '_CAM_Grayscale.png')
    save_image(Image.fromarray(np.repeat(np.expand_dims(
        cam_img*255, axis=2), 3, axis=2).astype(np.uint8)), path_to_file)


def save_gradient_img(orig_img, gradient, file_name):
    """
        Exports the original gradient image

    Args:
        gradient (np arr): Numpy array of the gradient with shape (3, 224, 224)
        file_name (str): File name to be exported
    """
    if not os.path.exists('./results'):
        os.makedirs('./results')

    orig_img = orig_img
    orig_img = np.transpose(orig_img, (1, 2, 0))
    # Normalize
    gradient = gradient - gradient.min()
    gradient /= gradient.max()

    gradient = np.transpose(gradient, (1, 2, 0))

    gradient = np.uint8(Image.fromarray(
        (gradient*255).astype(np.uint8)).resize(
        (orig_img.shape[1], orig_img.shape[0]), Image.ANTIALIAS))/255

    orig_img = Image.fromarray((orig_img*255).astype(np.uint8))

    grad_img = Image.fromarray((gradient*255).astype(np.uint8))
    # Save image
    path_to_file = os.path.join('./results', file_name + '.jpg')
    save_image(grad_img, path_to_file)

    # Overlay the heatmap and the image.
    gradient = np.pad(
        gradient, ((0, 0), (0, 0), (0, 1)), mode='constant',
        constant_values=0.8)
    grad_img = Image.fromarray((gradient*255).astype(np.uint8))

    colormap_on_image = Image.new("RGBA", orig_img.size)
    colormap_on_image = Image.alpha_composite(
        colormap_on_image, orig_img.convert('RGBA'))

    grad_on_image = Image.alpha_composite(
        orig_img.convert('RGBA'), grad_img)

    path_to_file = os.path.join('./results', file_name + '_Grad_on_Image.png')
    save_image(grad_on_image, path_to_file)


def save_image(img, path):
    '''
    Save the given PIL image to the specified path.

    Params:
        img (PIL image): Some image to be saved.
        path (string): Path to the save destination.

    Returns:
        None
    '''
    img.save(path)


def get_test_params(img_indx):
    """
    Collects all data that is used in the CNN visualizations.

    Args:
        img_indx (int): Specifies which image from the ImageNet
            validation set is returned.

    returns:
        orig_img (numpy array): Image from the ImageNet validation set
            with shape (D,W,H). No transformations are applied to this image.
        input_img (numpy_array): Image from the ImageNet validation set
            with shape (D,W,H). This image is normalized such that mean and
            variance are normalized.
        img_label (int): Correct class label of the image.
        model(Pytorch model): CNN model that is used for doing the tests.
    """
    # Read image
    orig_img, img_label = get_test_img(img_indx, normalized=False)
    # Process image
    input_img, _ = get_test_img(img_indx, normalized=True)
    # Load pretrained model.
    model = models.vgg16_bn(pretrained=True)
    return (orig_img,
            input_img,
            img_label,
            model)


def get_test_img(img_indx, normalized=True):
    """
    Get a test image from the ImageNet validation dataset.

    Args:
        img_indx (int): Specifies which image from the ImageNet
            validation set is returned.
        normalized (boolean): Defines whether normalization is applied to the
            image.

    returns:
        img (numpy array): Image from the ImageNet validation set with
            shape (D,W,H).
        label (int): Ground truth label of the image.
    """
    if normalized:
        # Use the standard normalization for ImageNet to load the images.
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
        ])
    else:
        # No transformations.
        transform = transforms.ToTensor()

    # Generate testloader.
    testset = datasets.ImageFolder(
        root='../data/ImageNet/', transform=transform)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=1, shuffle=False, num_workers=2)

    test_iter = iter(testloader)

    # Iterate to the given image index.
    for i in range(img_indx+1):
        img, label = test_iter.next()

    # Delete the batch dimension if not normalize since the image is not input
    # to a model in this case.
    if not normalized:
        img = img[0, :, :, :]

    return img, label
