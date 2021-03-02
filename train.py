import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.sampler import SubsetRandomSampler


from sklearn.model_selection import train_test_split
import pandas as pd
from PIL import Image
import os
from tqdm import tqdm
import cv2
from IPython.display import clear_output
import emoji


def accuracy(data_loader, net):
    correct = 0
    total = 0 

    with torch.no_grad():
        for data, labels in data_loader:
            outputs = net(data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100*correct/total
    return acc

def training(epochs, train_loader, val_loader, net, learning_rate = 0.003):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    train_log = []
    val_log = []
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):

    #         inputs, labels = data[0].to(device), data[1].to(device)
            inputs, labels = data

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        train_log.append(accuracy(train_loader, net))
        val_log.append(accuracy(val_loader, net))
            
        clear_output()
        print("Epoch", epoch)
        print("Train accuracy", train_log[-1])
        print("Val accuracy", val_log[-1])
        plt.plot(train_log, label='train accuracy')
        plt.plot(val_log, label='val accuracy')
        plt.legend(loc='best')
        plt.grid()
        plt.show()
            
    print('Finished Training')


def multi_class_accuracy(val_loader, net):
    classes = ('😁', '😳', '☹️', '😗', '🙄' , '😊', '😜')
    class_correct = list(0. for i in range(7))
    class_total = list(0. for i in range(7))
    with torch.no_grad():
        for i, data in enumerate(val_loader, 0):
            inputs, labels = data
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()

            for i in range(len(labels)):
              label = labels[i]
              class_correct[label] += c[i].item()
              class_total[label] += 1

    for i in range(7):
        string_to_print = 'Accuracy of ' + classes[i] + ': '
        p = class_correct[i]/class_total[i]
        print(string_to_print + '{:.2%}'.format(p))
