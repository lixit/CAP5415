from __future__ import print_function
import argparse
import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from torch.utils.tensorboard import SummaryWriter
from ConvNet import ConvNet 
import argparse
import matplotlib.pyplot as plt
import numpy as np


def show(img):
    img = img / 2 + 0.5   # unnormalize
    npimg = img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0))) 
    plt.show()


def train(model, device, train_loader, optimizer, criterion, epoch, batch_size, writer):
    '''
    Trains the model for an epoch and optimizes it.
    model: The model to train. Should already be in correct device.
    device: 'cuda' or 'cpu'.
    train_loader: dataloader for training samples.
    optimizer: optimizer to use for model parameter updates.
    criterion: used to compute loss for prediction and target 
    epoch: Current epoch. Only used for logging.
    batch_size: Batch size used. Only used for logging.
    '''
    
    # Set model to train mode before each epoch
    model.train()
    
    # Empty list to store losses 
    losses = []
    correct = 0
    
    # Iterate over entire training samples (1 epoch)
    for batch_idx, batch_sample in enumerate(train_loader):
        data, target = batch_sample

        # show(tv.utils.make_grid(data))
        # Push data/label to correct device
        data, target = data.to(device), target.to(device)
        
        # Reset optimizer gradients. Avoids grad accumulation (accumulation used in RNN).
        optimizer.zero_grad()
        
        # Do forward pass for current set of data
        output = model(data)
         
        # Compute loss based on criterion
        loss = criterion(output, target)
        
        # Computes gradient based on final loss
        loss.backward()
        
        # Store loss
        losses.append(loss.item())
        
        # Optimize model parameters based on learning rate and gradient 
        optimizer.step()
        
        # Get predicted index by selecting maximum log-probability
        pred = output.argmax(dim=1, keepdim=True)
        
        # Count correct predictions overall 
        correct += pred.eq(target.view_as(pred)).sum().item()
        
    train_loss = float(np.mean(losses))
    train_acc = correct / ((batch_idx+1) * batch_size)
   
    output = 'Train set: Epoch: {} Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format( epoch,
        train_loss, correct, (batch_idx+1) * batch_size, 100. * train_acc)
    writer.add_text('Train', output, epoch)
    print(output)
    return train_loss, train_acc
    


def test(model, device, test_loader, criterion, epoch, writer):
    '''
    Tests the model.
    model: The model to train. Should already be in correct device.
    device: 'cuda' or 'cpu'.
    test_loader: dataloader for test samples.
    '''
    
    # Set model to eval mode to notify all layers.
    model.eval()
    
    losses = []
    correct = 0
    
    # Set torch.no_grad() to disable gradient computation and backpropagation
    with torch.no_grad():
        for batch_idx, sample in enumerate(test_loader):
            data, target = sample
            data, target = data.to(device), target.to(device)

            # Predict for data by doing forward pass
            output = model(data)
            
            # Compute loss based on same criterion as training
            loss = criterion(output, target)
            
            # Append loss to overall test loss
            losses.append(loss.item())
            
            # Get predicted index by selecting maximum log-probability
            pred = output.argmax(dim=1, keepdim=True)
            
            # Count correct predictions overall 
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss = float(np.mean(losses))
    accuracy = 100. * correct / len(test_loader.dataset)
    output = 'Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset), accuracy)
    writer.add_text('Test', output, epoch)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))
    
    return test_loss, accuracy




def run_main(FLAGS):

    writer = SummaryWriter(FLAGS.log_dir)
    # Check if cuda is available
    use_cuda = torch.cuda.is_available()

    # Set proper device based on cuda availability 
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Torch device selected: ", device)
    
    # Initialize the model and send to device 
    model = ConvNet(FLAGS.mode).to(device)
    
    # nn.CrossEntropyLoss() combines nn.LogSoftmax() and nn.NLLLoss() in one single class.
    criterion = nn.CrossEntropyLoss()
    
    # Define optimizer function.
    optimizer = optim.SGD(model.parameters(), lr=FLAGS.learning_rate, momentum=0.9)
        
    
    # output of torchvision datasets are PILImage images of range [0, 1].
    # We transform them to Tensors of normalized range [-1, 1]
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    # Load datasets for training and testing
    # Inbuilt datasets available in torchvision (check documentation online)
    training_data = datasets.CIFAR10('./data/', train=True, download=True,
                       transform=transform)
    test_data = datasets.CIFAR10('./data/', train=False,
                       transform=transform)
    train_loader = DataLoader(training_data, batch_size = FLAGS.batch_size, 
                                shuffle=True, num_workers=4)
    test_loader = DataLoader(test_data, batch_size = FLAGS.batch_size, 
                                shuffle=False, num_workers=4)
    
    # dataiter = iter(train_loader)
    # images, labels = next(dataiter)
    # show(torchvision.utils.make_grid(images))
    
    # classes = ('plane', 'car', 'bird', 'cat',
    #        'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # # print labels
    # print(' '.join(f'{classes[labels[j]]:5s}' for j in range(FLAGS.batch_size)))

    best_accuracy = 0.0
    
    # Run training for n_epochs specified in config 
    for epoch in range(1, FLAGS.num_epochs + 1):
        train_loss, train_accuracy = train(model, device, train_loader,
                                            optimizer, criterion, epoch, FLAGS.batch_size, writer)
        test_loss, test_accuracy = test(model, device, test_loader, criterion, epoch, writer)
        
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
        print("epoch: {:d}, best test accuracy: {:2.2f}".format(epoch, best_accuracy))
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/test', test_loss, epoch)
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)
        writer.add_scalar('Accuracy/test', test_accuracy, epoch)
    
    torch.save(model.state_dict(), "cifar10.pt")
    print("accuracy is {:2.2f}".format(best_accuracy))
    
    print("Training and evaluation finished")
    
# CIFAR10 has 50000 training samples and 10000 test samples
# Each sample is 32x32x3
# python testCNN.py --mode 1 --learning_rate 0.001 --num_epochs 30 --batch_size 10 --log_dir log1
# python testCNN.py --mode 2 --learning_rate 0.001 --num_epochs 30 --batch_size 10 --log_dir log2
if __name__ == '__main__':
    # Set parameters for Sparse Autoencoder
    parser = argparse.ArgumentParser('CNN Exercise.')
    parser.add_argument('--mode',
                        type=int, default=1,
                        help='Select mode between 1-5.')
    parser.add_argument('--learning_rate',
                        type=float, default=0.1,
                        help='Initial learning rate.')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=60,
                        help='Number of epochs to run trainer.')
    parser.add_argument('--batch_size',
                        type=int, default=10,
                        help='Batch size. Must divide evenly into the dataset sizes.')
    parser.add_argument('--log_dir',
                        type=str,
                        default='logs',
                        help='Directory to put logging.')
    
    FLAGS = None
    FLAGS, unparsed = parser.parse_known_args()
    
    run_main(FLAGS)
    
    