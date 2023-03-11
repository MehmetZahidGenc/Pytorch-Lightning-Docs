from torchvision.datasets import ImageFolder
import pytorch_lightning as pl
import torch.utils.data as data
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms

class CustomData(pl.LightningDataModule):

    def __init__(self, data_dir, train_batch_size, val_batch_size, test_data=False):
        super(CustomData, self).__init__()
        self.data_dir = data_dir
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_data = test_data

        self.mnist_train = ImageFolder
        self.mnist_val = ImageFolder
        self.mnist_test = ImageFolder

        self.transform = transforms.Compose([transforms.ToTensor()])

    def prepare_data(self):
        # download
        datasets.MNIST(self.data_dir, train=True, download=True)
        datasets.MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage):
        mnist_full = datasets.MNIST(self.data_dir, train=True, transform=self.transform)
        self.mnist_train, self.mnist_val = data.random_split(mnist_full, [55000, 5000])

        self.mnist_test = datasets.MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.train_batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.val_batch_size)

    def test_dataloader(self):
        if self.test_data:
            return DataLoader(self.mnist_test, batch_size=self.val_batch_size)
        else:
            return DataLoader(self.mnist_val, batch_size=self.val_batch_size)