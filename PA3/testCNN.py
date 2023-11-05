from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from ConvNet import ConvNet 
import argparse
import numpy as np 
from collections import defaultdict
import matplotlib.pyplot as plt


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

        # Push data/label to correct device
        data, target = data.to(device), target.to(device)
        
        # Reset optimizer gradients. Avoids grad accumulation (accumulation used in RNN).
        optimizer.zero_grad()
        
        # Do forward pass for current set of data
        output = model(data)
         
        # Compute loss based on criterion
        loss = criterion(output, data)
        
        # Computes gradient based on final loss
        loss.backward()
        
        # Store loss
        losses.append(loss.item())
        
        # Optimize model parameters based on learning rate and gradient 
        optimizer.step()
        
        
    train_loss = float(np.mean(losses))
   
    output = 'Train set: Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss)
    writer.add_text('Train', output, epoch)
    print(output)
    return train_loss
    


def test(model, device, test_loader, writer):
    '''
    Tests the model.
    model: The model to train. Should already be in correct device.
    device: 'cuda' or 'cpu'.
    test_loader: dataloader for test samples.
    '''
    
    # Set model to eval mode to notify all layers.
    model.eval()

    digits = np.arange(10)
    counts = np.zeros(10)
    # dict of int to lists
    outputs = defaultdict(list)
    
    losses = []
    correct = 0
    
    # Set torch.no_grad() to disable gradient computation and backpropagation
    with torch.no_grad():
        for batch_idx, sample in enumerate(test_loader):
            data, target = sample
            data, target = data.to(device), target.to(device)

            output = model(data)
            
            for i in digits:
                if target.cpu().numpy() == i:
                    if counts[i] != 2:
                        outputs[i].append(data.reshape(28, 28).cpu().numpy())
                        outputs[i].append(model(data).reshape(28, 28).cpu().numpy())
                        counts[i] += 1
                    
                    # only one match possible, no need to compare the rest
                    continue
                
    f, axarr = plt.subplots(10, 4)

    # print the result
    for i in digits:
        print(type(outputs[i][0]))
        axarr[i, 0].imshow(outputs[i][0])
        axarr[i, 1].imshow(outputs[i][1])
        axarr[i, 2].imshow(outputs[i][2])
        axarr[i, 3].imshow(outputs[i][3])
        axarr[i, 0].set_title("input1")
        axarr[i, 1].set_title("output1")
        axarr[i, 2].set_title("input2")
        axarr[i, 3].set_title("input2")

    plt.show()
    return
    

def run_main(FLAGS):

    writer = SummaryWriter(FLAGS.log_dir)
    # Check if cuda is available
    use_cuda = torch.cuda.is_available()

    # Set proper device based on cuda availability 
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Torch device selected: ", device)
    
    # Initialize the model and send to device 
    model = ConvNet(FLAGS.mode).to(device)
    
    
    # Define loss function.
    criterion = nn.CrossEntropyLoss()
    if FLAGS.loss_function == 'MSELoss':
        criterion = nn.MSELoss()
    
    
    # Define optimizer function.
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=FLAGS.learning_rate, weight_decay=1e-5)
        
    
    # Create transformations to apply to each data sample 
    # Can specify variations such as image flip, color flip, random crop, ...
    transform=transforms.Compose([
        transforms.ToTensor() #,
        # transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    # Load datasets for training and testing
    # Inbuilt datasets available in torchvision (check documentation online)
    dataset1 = datasets.MNIST('./data/', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('./data/', train=False,
                       transform=transform)
    train_loader = DataLoader(dataset1, batch_size = FLAGS.batch_size, 
                                shuffle=True, num_workers=4)
    test_loader = DataLoader(dataset2, batch_size = 1, 
                                shuffle=False, num_workers=4)
    
    best_accuracy = 0.0
    
    if FLAGS.test:
        #Later to restore:
        model.load_state_dict(torch.load("model.pt"))
        test(model, device, test_loader, writer)
    else: 
        # Run training for n_epochs specified in config 
        for epoch in range(1, FLAGS.num_epochs + 1):
            train_loss = train(model, device, train_loader,
                                                optimizer, criterion, epoch, FLAGS.batch_size, writer)

            writer.add_scalar('Loss/train', train_loss, epoch)
        

        torch.save(model.state_dict(), "model.pt")
        print("Training finished")

    
    
    
    
# for train
# python testCNN.py --mode 6 --no-test --loss_function MSELoss --learning_rate 0.0001 --num_epochs 10 --batch_size 10 --log_dir log
# for test
# python testCNN.py --mode 6 --test --loss_function MSELoss --learning_rate 0.0001 --num_epochs 10 --batch_size 10 --log_dir log
if __name__ == '__main__':
    # Set parameters for Sparse Autoencoder
    parser = argparse.ArgumentParser('CNN Exercise.')
    parser.add_argument('--mode',
                        type=int, default=1,
                        help='Select mode between 1-5.')
    parser.add_argument('--test', action=argparse.BooleanOptionalAction,
                        type=bool, default=False,
                        help='train or test')
    parser.add_argument('--loss_function',
                        type=str, default='CrossEntropyLoss',
                        help='Loss Functoin. CrossEntropyLoss or MSELoss')
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
    
    