import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as pyplt


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

    # count the number of correctly classified samples per class
    if classlabels is not None:
        num_classes = torch.numel(classlabels)
        class_correct = list(0. for i in range(num_classes))
        class_total = list(0. for i in range(num_classes))

    # count the number of correctly classified samples compared to the total
    # number of samples
    correct = 0
    total = 0

    with torch.no_grad():
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
            class_accuracy[i] = class_correct[i] / class_total[i]
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

    # for act_map_indx in range(act_maps.size()[0]):
    #    act_maps[act_map_indx, :, :] = act_maps[act_map_indx, :, :]\
    #        / act_maps[act_map_indx, :, :].max()

    return images, labels, act_maps, predictions


def gen_heatmap_pure_gap(model, data):
    images, labels = data

    pred, act_maps = model.forward_dbg(images)

    return images, labels, act_maps, pred


def plot_heatmaps(images, labels, act_maps, predictions, num_maps):
    act_maps = act_maps.detach()

    act_maps = F.relu(act_maps)
    act_maps = act_maps / torch.max(act_maps)

    nrows = images.size()[0]
    ncols = min(6, predictions.size()[1])

    height = 6
    width = 8

    fig, axs = pyplt.subplots(
        nrows=nrows, ncols=ncols, figsize=[height, width])

    for ax in axs.flat:
        ax.axis('off')

    for img_indx in range(images.size()[0]):
        x = img_indx
        y = 0

        axs[x, y].imshow(
            images[img_indx, :, :, :].permute([1, 2, 0]).squeeze())
        axs[x, y].set_title(labels[img_indx])

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
                str(predictions[img_indx, sort_indcs[feature_indx].tolist()]))

    # pyplt.tight_layout(True)
    pyplt.show()
