import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


class CustomData(Dataset):
    def __init__(self, mask_root_dir, input_root_dir, input_channels, output_channels):
        super(CustomData, self).__init__()

        # to control the converting part of data loading
        self.input_channels = input_channels
        self.output_channels = output_channels

        # For Mask
        self.mask_root_dir = mask_root_dir
        self.list_files_mask = os.listdir(self.mask_root_dir)

        # For Input Image
        self.input_root_dir = input_root_dir
        self.list_files_input = os.listdir(self.input_root_dir)

        # Transformer for mask and input images
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.list_files_mask)

    def __getitem__(self, item):


        """ Mask Image """
        mask_image_file = self.list_files_mask[item]
        mask_image_path = os.path.join(self.mask_root_dir, mask_image_file)

        if self.output_channels == 1:
            mask_image = np.array(Image.open(mask_image_path).convert("L"), dtype=np.float32)
        elif self.output_channels == 3:
            mask_image = np.array(Image.open(mask_image_path).convert("RGB"))
        else:
            mask_image = np.array(Image.open(mask_image_path).convert("RGB")) # I'm not sure

        mask_image = Image.fromarray(mask_image)
        mask_image = self.transform(mask_image)



        """ Input Image """
        input_image_file = self.list_files_input[item]
        input_image_path = os.path.join(self.input_root_dir, input_image_file)

        if self.input_channels == 1:
            input_image = np.array(Image.open(input_image_path).convert("RGB")) # I'm not sure. Backbone has 3 input channel
        elif self.input_channels == 3:
            input_image = np.array(Image.open(input_image_path).convert("RGB"))
        else:
            input_image = np.array(Image.open(input_image_path).convert("RGB")) # I'm not sure

        input_image = Image.fromarray(input_image)
        input_image = self.transform(input_image)

        return input_image, mask_image


class SegmentationDataModule(pl.LightningDataModule):
    def __init__(self, input_channels, output_channels, train_mask_root_dir, train_input_root_dir, train_batch_size, val_mask_root_dir,
                 val_input_root_dir, val_batch_size):
        super(SegmentationDataModule, self).__init__()

        # for CustomData
        self.input_channels = input_channels
        self.output_channels = output_channels

        # For train data
        self.train_mask_root_dir = train_mask_root_dir
        self.train_input_root_dir = train_input_root_dir
        self.train_batch_size = train_batch_size

        # For validation data
        self.val_mask_root_dir = val_mask_root_dir
        self.val_input_root_dir = val_input_root_dir
        self.val_batch_size = val_batch_size

        self.train_dataset = ImageFolder
        self.val_dataset = ImageFolder

    def setup(self, stage):
        # Train dataset
        self.train_dataset = CustomData(mask_root_dir=self.train_mask_root_dir,
                                        input_root_dir=self.train_input_root_dir,
                                        input_channels=self.input_channels,
                                        output_channels=self.output_channels)

        # Val dataset
        self.val_dataset = CustomData(mask_root_dir=self.val_mask_root_dir,
                                      input_root_dir=self.val_input_root_dir,
                                      input_channels=self.input_channels,
                                      output_channels=self.output_channels)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.train_batch_size,
                                           shuffle=True, num_workers=2)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.val_batch_size,
                                           shuffle=False, num_workers=2)