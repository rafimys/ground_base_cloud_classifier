import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Dataset
import torchvision.transforms as transforms
import torchvision
import os
from tqdm.notebook import tqdm

import numpy as np
from PIL import Image, ImageDraw

import pandas as pd

class CloudImageTabularDataset(Dataset):
    def __init__(self, excel_path, image_path, transform=None):
        self.image_path = image_path
        self.transform = transform
        self.df = pd.read_excel(excel_path)
        self.tabular_cols = ['Temperature(℃)', 'Humidity(%RH)', 'Pressure(hpa)', 'Wind speed(m/s)']

    def set_transform(self, transform):
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # --- IMAGE ---
        img_path = os.path.join(self.image_path, f"{row['Name']}.jpg")
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.functional.to_tensor(image)

        # --- TABULAR ---
        tabular_data = row[self.tabular_cols].values.astype('float32')
        tabular = torch.from_numpy(tabular_data)

        # Twoja normalizacja tabelaryczna
        TAB_MEAN = torch.tensor([30.568, 55.4607, 1008.75, 0.913], dtype=torch.float32)
        TAB_STD  = torch.tensor([5.239, 16.976, 5.532, 0.863], dtype=torch.float32)
        tabular = (tabular - TAB_MEAN) / TAB_STD

        # --- LABEL ---
        # Używamy -1 jeśli Twoje klasy w nazwach są 1-7, a PyTorch wymaga 0-6
        y = torch.tensor(int(row['Name'].split('_')[0]) - 1, dtype=torch.long)

        return image, tabular, y