import torch
from torch import nn, tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Subset, ConcatDataset
from torch.utils.tensorboard import SummaryWriter

import lightning as L

from sklearn.model_selection import train_test_split

import torchvision
from torchvision import transforms
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

import gdown
import os
import copy
import os.path as osp
import zipfile

from gsn_classification.lightning_data_modules.tab import CloudImageTabularDataset

class CloudDataModule(L.LightningDataModule):
    def __init__(self,
                batch_size,
                data_dir: str = './MGCD/',
                train_dataset_path='./MGCD/MGCD/train',
                test_dataset_path='./MGCD/MGCD/test',
                num_workers=3,
                **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.train_dataset_path = train_dataset_path
        self.test_dataset_path = test_dataset_path
        self.num_workers = num_workers
        self.data_dir = data_dir
        self.zip_name = 'MGCD.zip'
        self.image_size = (252, 252)

        # --- AUGMENTACJA TRENINGOWA ---
        self.train_transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.RandomHorizontalFlip(p=0.5), # Odbicie lustrzane
            transforms.RandomVerticalFlip(p=0.2),   # Czasem chmury można odbić góra-dół
            transforms.RandomRotation(30),          # Mocniejszy obrót
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), # Przesunięcia
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2), # Mocniejsze zmiany światła
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # --- CZYSTE TRANSFORMACJE DLA VAL/TEST ---
        self.val_transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _build_dataset(self, root_dir, transform):
        datasets = []
        for file in sorted(os.listdir(root_dir)):
            if file.endswith(".xlsx"):
                excel_path = os.path.join(root_dir, file)
                folder_name = file.replace(".xlsx", "")
                image_path = os.path.join(root_dir, folder_name)
                if os.path.isdir(image_path):
                    datasets.append(CloudImageTabularDataset(excel_path, image_path, transform))
        return ConcatDataset(datasets)

    def _get_all_labels(self, dataset):
        return [dataset[i][2].item() for i in range(len(dataset))]

    def setup(self, stage=None):
        # 1. Zbiór treningowy - bierze wszystko z folderu train z silną augmentacją
        if stage in ("fit", None):
            self.train_dataset = self._build_dataset(self.train_dataset_path, self.train_transform)

            # 2. Zbiór testowy - ładujemy go, by wydzielić z niego walidację
            # Używamy val_transform, bo walidacja i test mają być "czyste"
            full_test_source = self._build_dataset(self.test_dataset_path, self.val_transform)

            # Pobieramy etykiety tylko dla źródła testowego
            test_labels = self._get_all_labels(full_test_source)

            # 3. Dzielimy zbiór testowy 50/50 na Walidację i Test
            val_idx, test_idx = train_test_split(
                range(len(full_test_source)),
                test_size=0.5,
                stratify=test_labels,
                random_state=42
            )

            self.val_dataset = Subset(full_test_source, val_idx)
            self.test_dataset = Subset(full_test_source, test_idx)

            print(f"--- KONFIGURACJA ZBIORÓW ---")
            print(f"Trening (z folderu train + AUG): {len(self.train_dataset)}")
            print(f"Walidacja (50% folderu test): {len(self.val_dataset)}")
            print(f"Test (50% folderu test): {len(self.test_dataset)}")


    def prepare_data(self):
        if not osp.isfile(self.zip_name): 
            gdown.download('https://drive.google.com/uc?id=17PBT__KLwJuAsUMSxnByGWaBkMaxGh4i', output=self.zip_name, quiet=False)
                            
        if not osp.isdir(self.data_dir):
            with zipfile.ZipFile(self.zip_name, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)