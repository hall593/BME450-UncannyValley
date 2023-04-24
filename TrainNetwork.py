import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import tensorflow as tf
from torchvision.io import read_image

tdataPath = r"C:/Users/macke/Desktop/450 Dataset/Training" # Change to specific file path for training folder 
tlabels = os.path.join(tdataPath, 'Labelled Data.csv') # label path for training

class CustomDataset(Dataset):
    # identifying
    def __init__(self, labels, imgDirectory, transform = None, target_transform = None):
        # Initializaation
        self.labels =  pd.read_csv(labels)
        self.imgDirectory = imgDirectory 
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        # Denotes total num of samples
        return len(self.labels)
    
    def __getitem__(self, index):
        # Generates one samples of data
        img_path = os.path.join(self.imgDirectory, self.labels.iloc[index, 0]) # error here 
        # above function works by joining path and image name for location
        image = read_image(img_path)
        imglabel = self.labels.iloc[index, 1] # reads second column for designated label

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            imglabel = self.target_transform(imglabel)
        return image, imglabel

def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result
    

class Rescale:
    def __init__(self, sizing = 200): 
        # if the sizing value is changed, values in Net() __init__ need to be changed for matrix multiplication
        self.sizing = sizing

    def __call__(self, tensor):
        toPIL = transforms.ToPILImage()(tensor) # converts tensor to image
        toPIL = toPIL.convert(mode='RGB') # ensures all pictures have same color channel
        toPIL.thumbnail((self.sizing, self.sizing)) # not sure if need, resizes keeping aspect ratio 
        resized = expand2square(toPIL, (0, 0, 0)).resize((self.sizing, self.sizing)) # adds padding and resizes again         
        newTen = transforms.ToTensor()(resized) # converts back to tensor
        return newTen


batch_size = 5

train_dataset = CustomDataset(tlabels, tdataPath, transform=Rescale())
train_dataloader = DataLoader(train_dataset, batch_size, shuffle = True)


train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(35344, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


# plotting loss with training and validation
t_loss = []
rounds = []

counter = 0
for epoch in range(20):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(train_dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        
        
        counter += 1
        if i % 5 == 4:    
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 5:.3f}')
            t_loss.append(running_loss)
            rounds.append(counter)
            running_loss = 0.0


print('Finished Training')
x = np.array(rounds)
y = np.array(t_loss)
plt.plot(x, y, color = 'orange')
plt.xlabel('Batch')
plt.ylabel('Loss')
plt.title('Loss per batch of training')
plt.show()

# Saving model
PATH = './cifar_net.pth' 
torch.save(net.state_dict(), PATH)

