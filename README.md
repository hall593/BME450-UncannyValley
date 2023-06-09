# BME450-UncannyValley
Training a neural network to recognize features of the different levels of uncanny valley

## Team members
Mackensie Hall (hall593), Joe Pierzakowski (ItAverageJoe/jpierzak)  

## Project Description
Using a neural network system, the goal of this project is to train it to identify different levels of the uncanny valley successfully. As the uncanny valley as a whole is a large scale, we will attempt to create a dataset that features a diverse range of examples. We will find as many examples as possible in order to ensure the accuracy of identification, along with assigning a value to signify how 'deep' an example is within the valley. Multiple parameters will be utilized and modified in ordered to influence the best performance of the network. Initially, we will start with a binary identification of whether an image is uncanny or not. If the neural network shows success, it will be trained to identify multiple categories.

The dataset used named "Full Dataset"

BaseNeuralNetwork is the code for the neural network itself.
The .pth files are saved, trained neural networks.
- cirfar_net.pth is the default path for the network trained on 200 by 200 px images
- sigmoid.pth is trained with sigmoid activation
- fivehund.pth is trained on 500 x 500 px images
