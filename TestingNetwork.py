
import torch
import os
from torch.utils.data import DataLoader
from TrainNetwork import Net, CustomDataset, Rescale, batch_size
import matplotlib.pyplot as plt

PATH = './cifar_net.pth'

net = Net()
net.load_state_dict(torch.load(PATH))

vdataPath = r"C:/Users/macke/Desktop/450 Dataset/Validation" # Change to specific file path for validation folder 
vlabels = os.path.join(vdataPath, 'Labelled Data.csv') # label path for validation

valid_dataset = CustomDataset(vlabels, vdataPath, transform=Rescale())
valid_dataloader = DataLoader(valid_dataset, batch_size, shuffle = True)


labels_map = { # can change to more extensive cateogorization if enough time, starting off with binary now
    0: "Not Uncanny",
    1: "Uncanny"
}


# count predictions overall
correct = 0
total = 0

# counting predictions for each class
correct_pred = {classname: 0 for classname in labels_map}
total_pred = {classname: 0 for classname in labels_map}


import torch.optim as optim
import torch.nn as nn
import numpy as np

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

t_loss = []
rounds = []

counter = 0

for epoch in range(20):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(valid_dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        index = 0
        images, labels = data
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(images)
        _, predictions = torch.max(outputs.data, 1)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        
        if i % 5 == 4:
            counter += 1    
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 5:.3f}')
            t_loss.append(running_loss)
            rounds.append(counter)
            running_loss = 0.0

        for label, prediction in zip(labels, predictions):

            if label == prediction:
                correct_pred[torch.IntTensor.item(label)] += 1
            '''
            else:
                test = images[index].permute(1,2,0).numpy() # changes tensor shape to readable image
                plt.imshow(test.squeeze(), cmap = "gray")
                Vdictvalue = torch.IntTensor.item(label) # v_labels returns tensor value, convert to int
                PdictValue = torch.IntTensor.item(prediction)
                print(f"Label: {[labels_map[Vdictvalue]]}")
                print(f"Prediction: {[labels_map[PdictValue]]}")
                plt.show()
            '''
            total_pred[torch.IntTensor.item(label)] += 1
            index += 1
        total += labels.size(0)
        correct += (predictions == labels).sum().item()


x = np.array(rounds)
y = np.array(t_loss)
plt.plot(x, y, color = 'orange')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Validation Loss')
plt.show()


print(f'Accuracy of the network on the given test images: {100 * correct // total} %')

# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname} is {accuracy:.1f} %')


v_images, v_labels = next(iter(valid_dataloader))
print(f"Feature batch shape: {v_images.size()}")

# printing out single image + label

'''
with torch.no_grad():
    for data in valid_dataloader: # validation data for testing
        index = 0 # number of images want to view + prediction
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predictions = torch.max(outputs.data, 1)
        for label, prediction in zip(labels, predictions):

            if label == prediction:
                correct_pred[torch.IntTensor.item(label)] += 1
            else:
                test = images[index].permute(1,2,0).numpy() # changes tensor shape to readable image
                plt.imshow(test.squeeze(), cmap = "gray")
                Vdictvalue = torch.IntTensor.item(label) # v_labels returns tensor value, convert to int
                PdictValue = torch.IntTensor.item(prediction)
                print(f"Label: {[labels_map[Vdictvalue]]}")
                print(f"Prediction: {[labels_map[PdictValue]]}")
                plt.show()
            total_pred[torch.IntTensor.item(label)] += 1
            index += 1
        total += labels.size(0)
        correct += (predictions == labels).sum().item()
'''
