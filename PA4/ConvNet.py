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
        # CIFAR-10 size 3x32x32

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1), # b, 64, 30, 30
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2), # b, 64, 15, 15
            nn.Conv2d(64, 192, kernel_size=3, stride=1), # b, 192, 13, 13
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2), # b, 192, 6, 6
            nn.Conv2d(192, 384, kernel_size=3, stride=1), # b, 384, 4, 4
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2) # b, 384, 2, 2
        )

        self.classifier1 = nn.Sequential(
            nn.Linear(384 * 2 * 2, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 10)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=5, stride=1), # b, 96, 28, 28
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2), # b, 96, 14, 14
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=1), # b, 256, 12, 12
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2), # b, 256, 6, 6
            nn.Conv2d(256, 384, kernel_size=5, stride=1, padding=1), # b, 384, 4, 4
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, stride=1), # b, 256, 2, 2
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2) # b, 256, 1, 1
        )

        self.classifier2 = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 10)
        )


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
            print("Invalid mode ", mode, "selected. Select between 1-2")
            exit(0)
        
        
    # Baseline model. step 1
    def model_1(self, X):
        x = self.conv1(X)
        x = self.classifier1(x.view(-1, 384 * 4 * 4))
        return x
    
    def model_2(self, X):
        x = self.conv2(X)
        x = self.classifier2(x.view(-1, 256))
        return x
    
