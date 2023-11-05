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

        # for autoencoder
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 256), # (28 * 28 + 1) * 256
            nn.ReLU(True),
            nn.Linear(256, 128), # (256 + 1) * 128
            nn.ReLU(True))
        self.decoder = nn.Sequential(
            nn.Linear(128, 256), # (128 + 1) * 256
            nn.ReLU(True),
            nn.Linear(256, 28*28), # (256 + 1) * 28 * 28
            nn.Tanh())
        
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 40, kernel_size=3, stride = 1,padding=1), # b, 40, 28, 28
            nn.MaxPool2d(2, stride=2), # b, 40, 14, 14
            nn.ReLU(True),
            nn.Conv2d(40, 49, kernel_size=3, stride = 1,padding=1), # b, 49, 14, 14
            nn.MaxPool2d(2, stride=2), # b, 49, 7, 7
            nn.ReLU(True))
        self.decoder_cnn = nn.Sequential(
            nn.Conv2d(49, 40, kernel_size=3, stride = 1,padding=1), # b, 40, 7, 7
            nn.Upsample(scale_factor=2, mode='bilinear'), # b, 40, 14, 14
            nn.ReLU(True),
            nn.Conv2d(40, 1, kernel_size=3, stride = 1,padding=1), # b, 1, 14, 14
            nn.Upsample(scale_factor=2, mode='bilinear'), # b, 1, 28, 28
            nn.ReLU(True),
            nn.Conv2d(1, 1, kernel_size=3, stride = 1,padding=1), # b, 1, 28, 28
            nn.Tanh())


        # # this use transpose convolution
        # self.decoder_cnn_transpose = nn.Sequential(
        #     nn.ConvTranspose2d(49, 40, kernel_size=3, stride = 2, padding=1), # b, 40, 13, 13
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(40, 1, kernel_size=3, stride = 2, output_padding=1), # b, 1, 28, 28
        #     nn.Tanh())

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
        elif mode == 6:
            self.forward = self.model_6
        elif mode == 7:
            self.forward = self.model_7
        else: 
            print("Invalid mode ", mode, "selected. Select between 6-7")
            exit(0)
        

    # autoencoder
    def model_6(self, X):
        x = self.encoder(X.view(-1, 28*28))
        # decoder
        x = self.decoder(x)
        return x.view(-1, 1, 28, 28)
    
    def model_7(self, X):
        x = self.encoder_cnn(X)
        x = self.decoder_cnn(x)
        return x