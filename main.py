import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from torch import nn, onnx
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
import seaborn as sns
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.transforms import transforms
from PIL import Image
import matplotlib.pyplot as plt

from EmotionNet import EmotionNet


def create_dataloader(dataset):
    # Split datasets 8:1:1
    train_indices = []
    valid_indices = []
    test_indices = []
    for class_index in range(len(dataset.classes)):
        class_indices = [i for i, (img, label) in enumerate(dataset.samples) if label == class_index]
        train_class_indices, test_class_indices = train_test_split(class_indices, test_size=0.2, random_state=42)
        valid_class_indices, test_class_indices = train_test_split(test_class_indices, test_size=0.5, random_state=42)

        train_indices.extend(train_class_indices)
        valid_indices.extend(valid_class_indices)
        test_indices.extend(test_class_indices)

    # Create subset dataloaders
    train_dataset = Subset(dataset, train_indices)
    valid_dataset = Subset(dataset, valid_indices)
    test_dataset = Subset(dataset, test_indices)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_dataloader, valid_dataloader, test_dataloader


def train_validation(train_model, train_dataset, valid_dataset, epochs, criterion):
    # load emotionNet model and set loss and optimizer
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # EmotionNet
    model = train_model.to(device)
    # loss
    criterion = criterion
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    # learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    # use tensorboard to record training process
    writer = SummaryWriter('./runs')
    ## train
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        # batch size
        for images, labels in train_dataset:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            # predicts
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicts = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicts == labels).sum().item()
        # update learning rate
        scheduler.step()
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, epochs, running_loss / len(train_dataloader)))
        # write outputs in tensorboard
        writer.add_scalar('training loss:', running_loss / len(train_dataloader), epoch)
        writer.add_scalar('training accuracy:', 100 * correct / total, epoch)
        # weight analysis
        for name, param in model.named_parameters():
            writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)

        ## validation
        model.eval()
        correct = 0
        total = 0
        val_predicts = []
        val_labels = []
        with torch.no_grad():
            for images, labels in valid_dataset:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicts = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicts == labels).sum().item()

                val_predicts.extend(predicts.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        # write outputs in tensorboard
        writer.add_scalar('validation loss:', running_loss / len(train_dataloader), epoch)
        writer.add_scalar('validation accuracy:', 100 * correct / total, epoch)

    # plot the outputs
    sns.heatmap(confusion_matrix(val_labels, val_predicts), annot=True, fmt='d', cmap='Pastel1')
    plt.xlabel('Validation Predicts')
    plt.ylabel('Validation Labels')
    plt.show()

    writer.close()
    # save model
    # torch.save(model.state_dict(), '/content/drive/MyDrive/comp8420/emotionNet_model.pth')
    dummy_input = torch.randn(1, 3, 224, 224, device=device)
    onnx_path = "./emotionNet.onnx"  # The file path to save the ONNX model
    onnx.export(model, dummy_input, onnx_path)

    return model


def test(test_model):
    correct = 0
    total = 0
    test_predicts = []
    test_labels = []

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = test_model.to(device)

    with torch.no_grad():
        for images, labels in test_dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicts = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicts == labels).sum().item()
            test_predicts.extend(predicts.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
    print('Test Accuracy: {:.2f}%'.format(100 * correct / total))
    # plot the outputs
    sns.heatmap(confusion_matrix(test_labels, test_predicts), annot=True, fmt='d', cmap='Pastel1', vmin=0, vmax=7)
    plt.xlabel('Test Predicts')
    plt.ylabel('Test Labels')
    plt.show()

    # get precision and recall
    precision = precision_score(test_labels, test_predicts, average=None)
    recall = recall_score(test_labels, test_predicts, average=None)

    return confusion_matrix(test_labels, test_predicts), precision, recall, model


def Grad_CAM(model):
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # last layer
    target_layers = [model.resnet.layer4[-1]]

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = Image.open(
        "/content/drive/MyDrive/comp8420/Subset For Assignment SFEW/Happy/Bridesmaids_000059880_00000039.png")
    input_tensor = transform(image).unsqueeze(0).to(device)

    # create cam
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=device == 'cuda')

    targets = [ClassifierOutputTarget(0)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
    visualization = show_cam_on_image(input_tensor.cpu().numpy()[0].transpose(1, 2, 0), grayscale_cam, use_rgb=True)

    plt.imshow(visualization)
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    """
        Step 1: Prepare dataloaders (train:validation:test = 8:1:1)
    """
    # data preprocess
    # resize image size and set it to tensor for dataloader
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    # use imagefolder to get all image samples with labels in each class folder
    dataset = datasets.ImageFolder(root='./Subset For Assignment SFEW',
                                   transform=transform)
    # get dataloaders
    train_dataloader, valid_dataloader, test_dataloader = create_dataloader(dataset)
    """
        Step 2: Start training and validation
    """
    model = train_validation(train_model=EmotionNet(),
                             train_dataset=train_dataloader,
                             valid_dataset=valid_dataloader,
                             epochs=50,
                             criterion=nn.CrossEntropyLoss())

    """
        Step 3: Test model, Precision, Recall and Specificity with Confusion matrix
    """
    # get confusion matrix, precision and recall
    cm, precision, recall, test_model = test(model)

    # specificity
    specificity = np.zeros_like(precision)
    for i in range(len(specificity)):
        true_negative = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
        false_positive = np.sum(cm[:, i]) - cm[i, i]
        specificity[i] = true_negative / (true_negative + false_positive)

    # show results
    for i in range(len(precision)):
        print("Class {}: Precision: {:.2f}, Recall: {:.2f}, Specificity: {:.2f}".format(i, precision[i], recall[i],
                                                                                        specificity[i]))
    """
        Step 4: Grad-CAM
    """
    Grad_CAM(test_model)