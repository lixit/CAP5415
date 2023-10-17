import time
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self, mode):
        super(ConvNet, self).__init__()
        
        # Define various layers here, such as in the tutorial example
        # self.conv1 = nn.Conv2D(...)
        # for MNISt dataset, the input size is 28x28
        self.fc1 = nn.Linear(28*28, 10)

        self.fc2 = nn.Linear(28*28, 100)
        self.fc3 = nn.Linear(100, 10)

        self.fc4 = nn.Linear(28*28, 1000)
        self.fc5 = nn.Linear(1000, 10)

        self.conv1 = nn.Conv2d(1, 40, 5, stride = 1)
        self.conv2 = nn.Conv2d(40, 49, 5, stride = 1)
        self.pool = nn.MaxPool2d(2, 2)
        # This will select the forward pass function based on mode for the ConvNet.
        # Based on the question, you have 5 modes available for step 1 to 5.
        # During creation of each ConvNet model, you will assign one of the valid mode.
        # This will fix the forward function (and the network graph) for the entire training/testing
        if mode == 1:
            self.forward = self.model_1
        elif mode == 2:
            self.forward = self.model_2
        elif mode == 3:
            self.forward = self.model_3
        elif mode == 4:
            self.forward = self.model_4
        elif mode == 5:
            self.forward = self.model_5
        else: 
            print("Invalid mode ", mode, "selected. Select between 1-5")
            exit(0)
        
        
    # Baseline model. step 1
    def model_1(self, X):
        # One fully connected layer.
        x = F.sigmoid(self.fc1(X.view(-1, 28*28)))
        return x

    # Use two convolutional layers.
    def model_2(self, X):
        # Two convolutional layers + one fully connnected layer.
        x = self.pool(F.sigmoid(self.conv1(X)))
        x = self.pool(F.sigmoid(self.conv2(x)))
        x = x.view(-1, 49 * 4 * 4)
        x = F.sigmoid(self.fc1(x))
        return x

    # Replace sigmoid with ReLU.
    def model_3(self, X):
        # Two convolutional layers + one fully connected layer, with ReLU.
        x = self.pool(F.relu(self.conv1(X)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 49 * 4 * 4)
        x = F.relu(self.fc1(x))
        return x

    # Add one extra fully connected layer.
    def model_4(self, X):
        # Two convolutional layers + two fully connected layers, with ReLU.
        x = self.pool(F.relu(self.conv1(X)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 49 * 4 * 4)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    # Use Dropout now.
    def model_5(self, X):
        # Two convolutional layers + two fully connected layers, with ReLU.
        # and  + Dropout.
        return NotImplementedError()
    
    
