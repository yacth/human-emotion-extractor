import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class EmotionDataset(Dataset):
    def __init__(self,csv_file, image_dir, data_type ,transform):
        
        self.csv_file = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.labels = self.csv_file['emotion']
        self.data_type = data_type
        self.transform = transform
          
    def __len__(self):
        return(len(self.csv_file))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_name = self.image_dir + self.data_type + str(idx) + '.jpg'
        image = Image.open(image_name)
        labels = np.array(self.labels[idx])
        labels = torch.from_numpy(labels).long()
        
        if self.transform:
            image = self.transform(image)
        return(image, labels)



