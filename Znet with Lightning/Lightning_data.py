import torch
from torchvision.datasets import ImageFolder
import pytorch_lightning as pl
from torchvision import datasets, transforms
import os

class CustomData(pl.LightningDataModule):

    def __init__(self, data_dir, train_batch_size, val_batch_size, test_data=False):
        super(CustomData, self).__init__()

        self.data_dir = data_dir
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_data = test_data

        self.train_image_dataset = ImageFolder
        self.val_image_dataset = ImageFolder
        self.test_image_dataset = ImageFolder

        self.train_data_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.val_data_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.CenterCrop(128),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.test_data_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.CenterCrop(128),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


    def setup(self, stage):

        self.train_image_dataset = datasets.ImageFolder(os.path.join(self.data_dir, 'train'), self.train_data_transform)
        train_data_size = len(self.train_image_dataset)

        class_names = self.train_image_dataset.classes

        self.val_image_dataset = datasets.ImageFolder(os.path.join(self.data_dir, 'val'), self.val_data_transform)
        val_data_size = len(self.val_image_dataset)

        if self.test_data:
            self.test_image_dataset = datasets.ImageFolder(os.path.join(self.data_dir, 'test'), self.test_data_transform)
            test_data_size = len(self.test_image_dataset)
        else:
            test_data_size = 0

        print(f'Train dataset sizes: {train_data_size}, Val dataset sizes: {val_data_size}, Test dataset sizes: {test_data_size}'
              f'\nClass names: {class_names}')

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_image_dataset, batch_size=self.train_batch_size,
                                           shuffle=True, num_workers=2)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_image_dataset, batch_size=self.val_batch_size,
                                           shuffle=False, num_workers=2)

    def test_dataloader(self):
        if self.test_data:
            return torch.utils.data.DataLoader(self.test_image_dataset, batch_size=self.val_batch_size,
                                               shuffle=True, num_workers=2)
        else:
            return torch.utils.data.DataLoader(self.val_image_dataset, batch_size=self.val_batch_size,
                                               shuffle=True, num_workers=2)