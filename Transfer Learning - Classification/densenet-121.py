import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import pytorch_lightning as pl
from torchvision.datasets import ImageFolder
import timm
import os
import torch.nn.functional as F
from torchmetrics import Accuracy, Recall, Precision, F1Score


"""
Lightning Module
"""

class ClassificationNet(pl.LightningModule):
    def __init__(self, num_classes):
        super(ClassificationNet, self).__init__()

        # Define num_classes
        self.num_classes = num_classes

        # Define metrics
        self.accuracy = Accuracy(task='multiclass', average='micro', num_classes=self.num_classes)
        self.precision_compute = Precision(task='multiclass', average='macro', num_classes=self.num_classes)
        self.recall_compute = Recall(task='multiclass', average='macro', num_classes=self.num_classes)
        self.f1 = F1Score(task='multiclass', average='macro', num_classes=self.num_classes)

        # Load the densenet121 pretrained model as the backbone
        self.backbone = timm.create_model('densenet121', pretrained=True)

        """
        "Freezing" a layer in a neural network means that the layer's parameters (i.e., weights and biases) are not updated during training. 
        In the context of transfer learning, we often use a pre-trained neural network as a starting point for a new task. 
        This is typically done by taking the pre-trained network and replacing the final layer(s) with a new layer(s) that is specific to the new task.

        When we freeze the layers in the pre-trained network, we prevent their weights and biases from being updated during training, so the network's existing knowledge is preserved. 
        In other words, we are telling the optimizer not to update the parameters of these layers during training.

        This is done because the initial layers of a pre-trained model usually learn more general features (such as edges and corners) that are useful for many tasks, while the later layers are more specific to the original task the model was trained on. 
        By freezing the early layers, we can use the pre-trained model as a feature extractor and only train the final layer(s) on the new task, which often leads to faster convergence and better performance.
        """

        # Freeze all the layers in the backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Replace the fully connected layer with a new one with the appropriate number of output classes
        # just change classifier layer to adjust for our own classes

        self.backbone.classifier = nn.Sequential(
                                   nn.Linear(in_features=1024, out_features=self.num_classes, bias=True),
                                )


    def forward(self, mzg):
        mzg = self.backbone(mzg)
        return mzg

    @staticmethod
    def cross_entropy_loss(y_hat, y):
        return F.cross_entropy(y_hat, y)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)

        # calculate metrics
        loss = self.cross_entropy_loss(y_hat=y_hat, y=y)
        accuracy = self.accuracy(y_hat, y)
        precision = self.precision_compute(y_hat, y)
        recall = self.recall_compute(y_hat, y)
        f1score = self.f1(y_hat, y)

        self.log_dict(
            {'train_loss': loss, 'train_accuracy': accuracy, 'train_precision': precision, 'train_recall': recall,
             'train_f1_score': f1score},
            on_step=False, on_epoch=True, prog_bar=True)

        return {'loss': loss, 'y': y, 'y_hat': y_hat, 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1 score': f1score}


    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)

        # calculate metrics
        loss = self.cross_entropy_loss(y_hat=y_hat, y=y)
        accuracy = self.accuracy(y_hat, y)
        precision = self.precision_compute(y_hat, y)
        recall = self.recall_compute(y_hat, y)
        f1score = self.f1(y_hat, y)

        self.log_dict(
            {'val_loss': loss, 'val_accuracy': accuracy, 'val_precision': precision, 'val_recall': recall,
             'val_f1_score': f1score},
            on_step=False, on_epoch=True, prog_bar=True)

        return {'loss': loss, 'y': y, 'y_hat': y_hat, 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1 score': f1score}



"""
Lightning Data Module
"""

class ClassificationDataset(pl.LightningDataModule):
    def __init__(self, dataset_root_dir, train_batch_size, val_batch_size, test_data=False, num_workers=2):
        super(ClassificationDataset, self).__init__()

        """
            Dataset Format;
            
            Dataset
              --> train
                    ->ClassName1
                          - imageX.jpg(or png)
                    ->ClassName2
                          - imageY.jpg(or png)
                          
              --> val
                    same architecture as train
                    
              --> test(optional)
                    same architecture as train
        """

        self.data_root_dir = dataset_root_dir
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_data = test_data

        # Test data is optional generally
        if self.test_data:
            self.test_batch_size = self.val_batch_size

        # Define ImageFolder variables to use in setup function
        self.train_image_dataset = ImageFolder
        self.val_image_dataset = ImageFolder
        self.test_image_dataset = ImageFolder

        # number of workers
        self.num_workers = num_workers


        # Define transformer for each dataset part
        self.train_data_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

        self.val_data_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.CenterCrop(128),
            transforms.ToTensor()
        ])

        self.test_data_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.CenterCrop(128),
            transforms.ToTensor()
        ])


    def setup(self, stage):
        # Fill the dataset ImageFolder variables
        self.train_image_dataset = datasets.ImageFolder(os.path.join(self.data_root_dir, 'train'), self.train_data_transform)

        self.val_image_dataset = datasets.ImageFolder(os.path.join(self.data_root_dir, 'val'), self.val_data_transform)

        if self.test_data:
            self.test_image_dataset = datasets.ImageFolder(os.path.join(self.data_root_dir, 'test'),
                                                           self.test_data_transform)
            test_data_size = len(self.test_image_dataset)
        else:
            test_data_size = 0


        # Info about datasets
        train_data_size = len(self.train_image_dataset)
        class_names = self.train_image_dataset.classes
        val_data_size = len(self.val_image_dataset)


        # Give some information about dataset sizes and class names
        print(
            f'Train dataset size: {train_data_size}, '
            f'Val dataset size: {val_data_size}, '
            f'Test dataset size: {test_data_size}'
            f'\nClass names: {class_names}'
        )

    # These 3 functions load the datasets
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_image_dataset, batch_size=self.train_batch_size,
                                           shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_image_dataset, batch_size=self.val_batch_size,
                                           shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        if self.test_data:
            return torch.utils.data.DataLoader(self.test_image_dataset, batch_size=self.val_batch_size,
                                               shuffle=True, num_workers=self.num_workers)
        else:
            return torch.utils.data.DataLoader(self.val_image_dataset, batch_size=self.val_batch_size,
                                               shuffle=True, num_workers=self.num_workers)



"""
Let's train the model with our custom dataset
"""

# create model
model = ClassificationNet(num_classes=5)

# create lightning datamodule
data_module = ClassificationDataset(dataset_root_dir='Chicken', train_batch_size=16, val_batch_size=16, test_data=False, num_workers=0)

# Trainer / you can configure trainer
trainer = pl.Trainer(max_epochs=5)

# fitting
trainer.fit(model, data_module)