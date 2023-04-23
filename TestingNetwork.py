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

with torch.no_grad():
    for data in valid_dataloader: # validation data for testing
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predictions = torch.max(outputs.data, 1)
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[torch.IntTensor.item(label)] += 1
            total_pred[torch.IntTensor.item(label)] += 1
        total += labels.size(0)
        correct += (predictions == labels).sum().item()


print(f'Accuracy of the network on the given test images: {100 * correct // total} %')

# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname} is {accuracy:.1f} %')


v_images, v_labels = next(iter(valid_dataloader))
print(f"Feature batch shape: {v_images.size()}")

# printing out single image + label
test = v_images[0].permute(1,2,0).numpy() # changes tensor shape to readable image
plt.imshow(test.squeeze(), cmap = "gray")
dictvalue = torch.IntTensor.item(v_labels[0]) # v_labels returns tensor value, convert to int
print(f"Label: {[labels_map[dictvalue]]}")
plt.show()
