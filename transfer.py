import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from torchvision import datasets, models, transforms
import time
import os
import copy

# This Python script performs transfer learning on the ResNet18 model
# adapting it to perform classification on hotwheels cars

device = torch.device("cpu")    # i don't have a skibidi nvidia gpu


# ResNet18 is trained on ImageNet1k, and expects images to normalize their size and RGB values
# normalized value = (pixel value - mean) / standard deviation
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),  # resize new images to a random grid of 224 x 224 pixels (reduce overfitting)
        transforms.RandomHorizontalFlip(),  # possibly flip the image with 50% probability (reduce overfitting)
        transforms.ToTensor(),              # convert image to a PyTorch tensor
        transforms.Normalize(mean, std)     # Normalize RGB channels
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),             # a square of 256 x 256 pixels
        transforms.CenterCrop(224),         # take a grid of 224 x 224 from the center of the 256 x 256 image
        transforms.ToTensor(),              # same
        transforms.Normalize(mean, std)     # same
    ])
}

# import data
data_dir = './data'
sets = ['train', 'val']
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), 
                                          data_transforms[x]) 
                  for x in ['train', 'val']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, 
                                              shuffle=True, num_workers=0) 
               for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
print(class_names)

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        # Are we randomly shuffling the training data for each epoch?
        print(f"Epoch {epoch}/{num_epochs-1}")
        print("-" * 10)

        # for every epoch, we have a training and validation phase
        for phase in ['train', 'val']:  
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print("{} Loss: {:.4f} Accuracy: {:.4f}".format(phase, epoch_loss, epoch_acc))

            # deep copy model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print("Training complete in {:.0f}m {:0f}s".format(time_elapsed // 60, time_elapsed % 60))
    print("Best val Accuracy: {:.4f}".format(best_acc))

    # load best model weights 
    model.load_state_dict(best_model_wts)
    return model

# Fine tuning of whole model
# model = models.resnet18(pretrained=True)
# num_features = model.fc.in_features # number of features of the last input layer (fc means fully connected)

# # we add a new output layer, it takes in the same number of inputs but gives 5 outputs, 
# # one for each hot wheel model (P1, GTR, S, Y, Huracan)
# model.fc = nn.Linear(num_features, 5)
# model.to(device)

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.001)   # stocastic gradient descent, standard learning rate

# # scheduler
# # every 7 epochs the learning rate is multiplied by the gamma
# step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# # Fine-tuned model
# model = train_model(model, criterion, optimizer, step_lr_scheduler, num_epochs=20)


# Freeze all layers except the last and train that one
model = models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad(False)

num_features = model.fc.in_features # number of features of the last input layer (fc means fully connected)

# we add a new output layer, it takes in the same number of inputs but gives 5 outputs, 
# one for each hot wheel model (P1, GTR, S, Y, Huracan)
model.fc = nn.Linear(num_features, len(class_names))
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)   # stocastic gradient descent, standard learning rate

# scheduler
# every 7 epochs the learning rate is multiplied by the gamma
step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Fine-tuned model
model = train_model(model, criterion, optimizer, step_lr_scheduler, num_epochs=20)


# add an “unknown/not a car” class or reject predictions below a calibrated confidence threshold.
# open up the webcam and classify
# shuffle the training data images in each epoch