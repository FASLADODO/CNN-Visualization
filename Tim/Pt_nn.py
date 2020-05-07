import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import cv2

import matplotlib.pyplot as plt
import matplotlib.image as mpimage

import ImageNet_classlabels


# train the neural network to classify images of the CIFAR10 dataset.
def train_network(
        model,
        optimizer,
        loss_func=nn.CrossEntropyLoss(),
        num_epochs=2,
        trainloader=None,
        validloader=None,
        loadpath=None,
        savepath='./data/models/model.pth'):

    best_accuracy = 0

    # train the network for 'num_epochs' epochs
    for epoch in range(num_epochs):

        model.train()       # set to training mode

        # iterate over the full training set
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):

            inputs, labels = data

            # reset the accumulated gradients
            optimizer.zero_grad()

            # Perform one optimization step with forward pass, backward
            # pass loss calculation and optimization.
            output = model(inputs)
            loss = loss_func(output, labels)
            loss.backward()
            optimizer.step()

            # Accumulate loss and print the average loss every 100
            # iterations.
            running_loss += loss.item()
            if i % 100 == 99:
                print(
                    '[%d, %5d] loss: %.3f'
                    % (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

        accuracy, _ = test_classification_accuracy(
            model, validloader)

        # save the best performing network to memory
        if(accuracy > best_accuracy and savepath is not None):
            torch.save(model.state_dict(), savepath)

    print('Finished Training')

    return model


# test the classification accuracy of a given model
def test_classification_accuracy(
        model,
        testloader,
        classlabels=None):

    model.eval()

    # count the number of correctly classified samples per class
    if classlabels is not None:
        num_classes = len(classlabels)
        class_correct = list(0. for i in range(num_classes))
        class_total = list(0. for i in range(num_classes))

    # count the number of correctly classified samples compared to the total
    # number of samples
    correct = 0
    total = 0

    for data in testloader:
        input, labels = data
        outputs = model(input)
        # total prediction accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # per class prediction accuracy
        if classlabels is not None:
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(torch.numel(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    class_accuracy = []
    if classlabels is not None:
        for i in range(num_classes):
            class_accuracy.append(class_correct[i] / class_total[i])
            print('Accuracy of %10s : %2d %%' % (
                classlabels[i], 100 * class_accuracy[i]))

    total_accuracy = correct / total
    print(
        'Accuracy of the network on the test images: %.2f %%'
        % (100 * total_accuracy))

    return total_accuracy, class_accuracy


def gen_heatmap_gap(model, data):
    images, labels = data

    pred, x_act = model.forward_dbg(images)

    # Compute the predicted class.
    predictions = F.softmax(pred, 1)

    # Get the weights from the activation layer.
    weights = model.fc.weight

    # Multiply the activation maps with the activation weights.
    act_maps = torch.zeros(
        x_act.size()[0],    # number of samples per batch
        weights.size()[0],    # number of classes
        x_act.size()[1],    # number of features maps
        x_act.size()[2],    # x-dim. of feature map
        x_act.size()[3])    # y-dim. of feature map

    for img_indx in range(x_act.size()[0]):
        for class_indx in range(act_maps.size()[1]):
            for feature_map in range(weights.size(1)):
                act_maps[img_indx, class_indx, feature_map, :, :] = \
                    x_act[img_indx, feature_map, :, :]\
                    * weights[class_indx, feature_map]

    act_maps = act_maps.sum(2)

    return images, labels, act_maps, predictions


def gen_heatmap_pure_gap(model, data):
    images, labels = data

    pred, act_maps = model.forward_dbg(images)

    return images, labels, act_maps, pred


# Generate the heatmap and overlays it over the given image.
#   model:          Neural network model
#   image:          Input image to classify
#   class_indx:     Index of class for which the activation map should be
#                   generated
#   feature_layer:  The layer to which the CAM belongs to
#   filename:       Name of the file that the CAM is saved to
def gen_heatmap_grad(
        model,
        input_image,
        image,
        class_index=None,
        feature_layers=[0],
        filename='CAM'):

    # Make forward pass through the network with the given set of images.
    model.eval()

    model.reset()

    pred = model(input_image)

    # Get the predicted class.
    if class_index is None:
        _, class_index = torch.max(pred, 1)
    # predictions = F.softmax(pred, 1)

    print(ImageNet_classlabels.classlabels[class_index.item()])

    # Backward pass to compute the gradients of the activation layers.
    pred[:, class_index].backward()

    imgs = []

    for feature_layer in feature_layers:

        # Get the gradients from the model.
        gradients = model.get_feature_gradients(feature_layer)
        # Compute the weighting coefficients by averaging over the gradients of
        # each feature layer.
        alpha = torch.mean(gradients, dim=[0, 2, 3])

        # Get the activation maps from the model.
        activations = model.get_feature_maps(
            input_image, feature_layer).detach()

        # Weight the activation maps with the weighting coefficients alpha.
        for i in range(activations.size()[1]):
            activations[:, i, :, :] *= alpha[i]

        # Average the activations over all channels
        heatmap = torch.mean(activations, dim=1).squeeze()

        # Apply ReLU to the heatmap in order to only account for positive
        # contributions to the predicted class.
        heatmap = F.relu(heatmap)

        # Normalize the heatmap
        heatmap = heatmap / torch.max(heatmap)

        # Convert image tensor to three channel image
        if image.shape[0] == 3:
            image_rgb = image
        else:
            image_rgb = image.repeat(3, 1, 1)

        image_rgb = image_rgb.permute(1, 2, 0) * 255

        heatmap = cv2.resize(heatmap.numpy(), (image.shape[2], image.shape[1]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = heatmap * 0.4 + np.uint8(image_rgb.numpy())

        filepath = './data/FeatureMaps/' + filename +\
            '_c' + str(class_index.item()) +\
            '_f' + str(feature_layer) +\
            '.jpg'

        cv2.imwrite(filepath, superimposed_img)

        # clip image to range [0, 255]
        superimposed_img = np.maximum(0, superimposed_img)
        superimposed_img = np.minimum(255, superimposed_img)

        img = mpimage.imread(filepath)

        _, pred_indx = torch.max(pred, 1)

        imgs.append(img)

    return imgs, pred_indx, F.softmax(pred, 1).squeeze()[pred_indx]


def plot_heatmaps(images, labels, act_maps, predictions, num_maps):
    act_maps = act_maps.detach()

    act_maps = F.relu(act_maps / torch.max(act_maps))

    nrows = images.size()[0]
    ncols = min(6, predictions.size()[1])

    height = 6
    width = 8

    fig, axs = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=[height, width])

    for ax in axs.flat:
        ax.axis('off')

    for img_indx in range(images.size()[0]):
        x = img_indx
        y = 0

        axs[x, y].imshow(
            images[img_indx, :, :, :].permute([1, 2, 0]).squeeze())
        axs[x, y].set_title(labels.numpy()[img_indx])

    for img_indx in range(images.size()[0]):
        sort_indcs = torch.argsort(
            predictions[img_indx, :], descending=True)
        for feature_indx in range(ncols - 1):
            x = img_indx
            y = feature_indx + 1

            axs[x, y].imshow(
                act_maps[img_indx, sort_indcs[feature_indx], :, :].squeeze(),
                cmap='gray')
            axs[x, y].set_title(
                str(sort_indcs[feature_indx].tolist()) + ": " +
                str(predictions.detach().numpy()[
                    img_indx, sort_indcs[feature_indx].tolist()]))

    plt.show()
