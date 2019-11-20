# Assignment 07

```
Build a binary classifier based on fully connected layers for the human versus horse dataset using pytorch library 
```

## Binary classification based on fully connected neural network

## Neural Network Architecture

- Build a neural network model based on the fully connected layers with pytorch library
- You can determine the number of layers
- You can determine the size of each layer
- You can determine the activation function at each layer except the output layer
- You use the sigmoid function for the activation fuction at the output layer

## Loss function with a regularization term based on $`L_2^2`$ norm

$`\mathcal{L} = \frac{1}{n} \sum_{i=1}^n f_i + \frac{\lambda}{2} \left( \| W \|_2^2 \right)`$

- Cross Entropy : 
    $`f_i = - y_i \log \hat{y}_i - (1 - y_i) \log (1 - \hat{y}_i) `$, where $`y_i`$ denotes the ground truth and $`\hat{y}_i`$ denotes the output of the network

- Regularization : 
    $`\| W \|_2^2 = \left( \sum_i w_{i}^2 \right)`$, where $`w_{i}`$ denotes all the model parameters

## Dataset

- The dataset consists of human images and horse images for the training and the validation
- The classifier should be trained using only the training set
- The classifier should be tested using only the validation set

## Implementation

- Write codes in python programming
- Use ```jupyter notebook``` for the programming environment
- You should use pytorch library for the construction of the model and the optimization

### Neural Network Model in pytorch (Linear.py)

```python
import torch.nn as nn
import torch.nn.functional as F
import math

class Linear(nn.Module):

    def __init__(self, num_classes=2):

        super(Linear, self).__init__()

        self.number_class   = num_classes

        _size_image     = 100* 100
        _num1           = 50
        _num2           = 50
        
        self.fc1        = nn.Linear(_size_image, _num1, bias=True)
        self.fc2        = nn.Linear(_num1, _num2, bias=True)
        self.fc3        = nn.Linear(_num2, num_classes, bias=True)

        self.fc_layer1  = nn.Sequential(self.fc1, nn.ReLU(True))
        self.fc_layer2  = nn.Sequential(self.fc2, nn.ReLU(True))
        self.fc_layer3  = nn.Sequential(self.fc3, nn.ReLU(True))
        
        self.classifier = nn.Sequential(self.fc_layer1, self.fc_layer2, self.fc_layer3)
        
        self._initialize_weight()        
        
    def _initialize_weight(self):

        for m in self.modules():
            
            n = m.in_features
            m.weight.data.uniform_(- 1.0 / math.sqrt(n), 1.0 / math.sqrt(n))

            if m.bias is not None:

                m.bias.data.zero_()

    def forward(self, x):

        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x
```

### Training and Testing in pytorch (main.py)

```python
# -----------------------------------------------------------------------------
# import packages
# -----------------------------------------------------------------------------
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import argparse
import sys
import os
import numpy as np
import time
import datetime 
import csv
import configparser
import argparse
import platform

from torchvision import datasets, transforms
from torch.autograd import Variable
from random import shuffle

# -----------------------------------------------------------------------------
# load dataset
# -----------------------------------------------------------------------------

set_train   = 
set_test    = 

num_classes = 2

# -----------------------------------------------------------------------------
# load neural network model
# -----------------------------------------------------------------------------

from Linear import *
    model = Linear(num_classes=num_classes)

# -----------------------------------------------------------------------------
# Set the flag for using cuda
# -----------------------------------------------------------------------------

bCuda = 1

if bCuda:
 
    model.cuda()

# -----------------------------------------------------------------------------
# optimization algorithm
# -----------------------------------------------------------------------------

optimizer   = optim.SGD(model.parameters())
objective   = nn.CrossEntropyLoss()

# -----------------------------------------------------------------------------
# function for training the model
# -----------------------------------------------------------------------------

def train():

    # print('train the model at given epoch')

    loss_train          = []

    model.train()

    for idx_batch, (data, target) in enumerate(loader_train):

        if bCuda:
        
            data, target    = data.cuda(), target.cuda()

        data, target    = Variable(data), Variable(target)

        optimizer.zero_grad()

        output  = model(data)
        loss    = objective(output, target)

        loss.backward()
        optimizer.step()

        loss_train_batch    = loss.item() / len(data)
        loss_train.append(loss_train_batch)
        
    loss_train_mean     = np.mean(loss_train)
    loss_train_std      = np.std(loss_train)

    return {'loss_train_mean': loss_train_mean, 'loss_train_std': loss_train_std}

# -----------------------------------------------------------------------------
# function for testing the model
# -----------------------------------------------------------------------------

def test():

    # print('test the model at given epoch')

    accuracy_test   = []
    loss_test       = 0
    correct         = 0

    model.eval()

    for idx_batch, (data, target) in enumerate(loader_test):

        if bCuda:
        
            data, target    = data.cuda(), target.cuda()

        data, target    = Variable(data), Variable(target)

        output  = model(data)
        loss    = objective(output, target)

        loss_test   += loss.item()
        pred        = output.data.max(1)[1]
        correct     += pred.eq(target.data.view_as(pred)).cpu().sum()

    loss_test       = loss_test / len(loader_test.dataset)
    accuracy_test   = 100. * float(correct) / len(loader_test.dataset)

    return {'loss_test': loss_test, 'accuracy_test': accuracy_test}

# -----------------------------------------------------------------------------
# iteration for the epoch
# -----------------------------------------------------------------------------

for e in range(epoch):
        
    result_train    = train()
    result_test     = test()

    loss_train_mean[e]  = result_train['loss_train_mean']
    loss_train_std[e]   = result_train['loss_train_std']
    loss_test[e]        = result_test['loss_test']
    accuracy_test[e]    = result_test['accuracy_test']
```


## Optimization

- You can use weight decay option in the pytorch optimization function
- You can use mini-batch gradient descent (stochastic gradient descent) with your choice of mini-batch size
- You can use a different learning rate at each iteration
- You can initialize the values of the model parameters with your choice of algorithm
- You should apply enough number of iterations that lead to the convergence of the algorithm

## git commit

- Apply a number of ```git commit``` at intermediate development steps with their descriptive comments 

#### Output (text)

- Print out the followings at each epoch
    - average training loss within the mini-batch cross iterations in the training data
    - average training accuracy within the mini-batch cross iterations in the training data
    - testing loss using the testing data at each epoch
    - testing accracy using the testing data at each epoch
    - [epoch #####] loss: (training) #####, (testing) #####, accuracy: (training) #####, (testing) #####

#### Output (graph)

- Plot the average of the training loss within the mini-batch cross iterations
- Plot the standard deviation of the training loss withint the mini-batch cross iterations
- Plot the average of the training accuracy within the mini-batch cross iterations
- Plot the standard deviation of the training accuracy withint the mini-batch cross iterations
- Plot the testing loss at each epoch
- Plot the testing accuracy at each epoch

#### Output (table)

- Present the final loss and accuracy at convergence

| dataset    | loss       | accuracy   |
|:----------:|:----------:|:----------:|
| training   |            |            |
| validation |            |            |

## Grading

- The grading is given by the validation accuracy for the best generalization (10 digits after the decimal point)
- top 50% would get the score 5 and bottom 50% would get the score 4 (only complete submissions will be considered)
- The maximum score for incomplete submissions will be the score 3

## Submission

- A PDF file exported from jupyter notebook for codes, results and comments
- A PDF file exported from the github website for the history of git commit








