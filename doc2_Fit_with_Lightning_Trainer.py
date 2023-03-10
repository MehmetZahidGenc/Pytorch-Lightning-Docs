import pytorch_lightning as pl
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from doc1_Define_LightningModule import AENet

ae_model = AENet(in_channel=1)

dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
train_loader = DataLoader(dataset)

trainer = pl.Trainer()
trainer.fit(ae_model, train_loader)