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


def csv_to_images(csv_entry):
    image_vector = csv_entry.split(' ')
    image_array = np.asarray(image_vector, dtype=np.uint8).reshape(48, 48)
    return Image.fromarray(image_array)


def save_images(dataset_path, data_type):
    data_csv_path = dataset_path + '/' + data_type + '.csv'
    data_folder_name = dataset_path + '/' + data_type
    
    if not os.path.exists(data_folder_name):
        os.mkdir(data_folder_name)
    
    data = pd.read_csv(data_csv_path)
    images = data['pixels']
    nb_of_images = images.shape[0]
    for idx in tqdm(range(nb_of_images)):
        image = csv_to_images(images[idx])
        image.save(os.path.join(data_folder_name, '{}{}.jpg'.format(data_type, idx)), 'JPEG')
    print('Saved {} data'.format((data_folder_name)))
    
    
def train_val_split_data(dataset_path):
    actual_train_data = pd.read_csv(dataset_path + '/' + 'train.csv')
    train_data, val_data = train_test_split(actual_train_data, test_size=0.33, random_state=123)
    train_data.to_csv(dataset_path + '/' + 'train_data.csv')
    val_data.to_csv(dataset_path + '/' + 'val_data.csv')    
    
def imshow(img, label):
    print(f"Label: {label}") 
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

