import torch
import pytorch_lightning as pl
import timm
import torch.nn.functional as F
import torch.nn as nn


class SegmentationModel(pl.LightningModule):
    def __init__(self, input_channels, output_channels, learning_rate=1e-3):
        super().__init__()

        self.learning_rate = learning_rate
        self.input_channels = input_channels
        self.output_channels = output_channels

        # it will be used to control saving image
        self.number = 0

        # Load the model as backbone
        self.backbone = timm.create_model('vgg19', pretrained=True)

        # Delete classifier(or fc, actually last part of network) part of backbone. This part occurs an error about dimension of tensor
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])

        self.decoder = nn.Sequential(
            # in channel, "4096", according to output layers of configured backbone
            nn.ConvTranspose2d(4096, 2048, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(2048, 1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, self.output_channels, kernel_size=1),
        )

    def forward(self, x):
        feats = self.backbone(x)

        x = self.decoder(feats)

        return x

    @staticmethod
    def intersection_over_union(y_hat, y):
        intersection = torch.sum(y_hat == y)
        union = torch.sum((y_hat > 0)+(y > 0))
        return intersection / union

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)

        loss = F.mse_loss(y_hat, y)
        iou = self.intersection_over_union(y_hat, y)

        self.log_dict({'train loss': loss, 'train iou loss': iou})
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)

        loss = F.mse_loss(y_hat, y)
        iou = self.intersection_over_union(y_hat, y)

        self.log_dict({'val loss': loss, 'val iou loss': iou})
        return loss