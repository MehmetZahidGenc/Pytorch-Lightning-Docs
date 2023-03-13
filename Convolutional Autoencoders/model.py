import torch
import pytorch_lightning as pl
import torch.nn as nn
from torchvision.utils import save_image

"""
        Convolutional Auto Encoders with Lightning AI
                         __________
        Input Image -->  | ConvAE |  --> Target Image 
                         |________|
"""

class ConvAE(pl.LightningModule):
    def __init__(self, in_channel, features=[16, 32, 64]):
        super(ConvAE, self).__init__()
        self.features = features
        self.in_channel = in_channel

        self.criterion = nn.MSELoss()

        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()

        self.number = 0 # it will be used to control test prediction result saving

        # encoder
        self.layers_encoder = []
        for i in range(len(self.features)):
            if i == 0:
                self.layers_encoder.append(nn.Conv2d(in_channel, features[i], kernel_size=3, stride=2, padding=1))
                self.layers_encoder.append(nn.ReLU())

            else:
                if i != len(self.features)-1:
                    self.layers_encoder.append(
                        nn.Conv2d(features[i-1], features[i], kernel_size=3, stride=2, padding=1))
                    self.layers_encoder.append(nn.ReLU())
                elif i == len(self.features)-1:
                    self.layers_encoder.append(nn.Conv2d(features[i-1], features[i], kernel_size=7))

        # decoder
        self.layers_decoder = []
        self.features = self.reversList(self.features)
        for j in range(len(self.features)):
            if j == len(self.features)-1:
                self.layers_decoder.append(
                    nn.ConvTranspose2d(self.features[j], in_channel, kernel_size=3, stride=2, padding=1,
                                       output_padding=1))
                self.layers_decoder.append((nn.Sigmoid()))
            else:
                if j == 0:
                    self.layers_decoder.append(
                        nn.ConvTranspose2d(self.features[j], self.features[j+1], kernel_size=7))
                    self.layers_decoder.append(nn.ReLU())
                else:
                    self.layers_decoder.append(
                        nn.ConvTranspose2d(self.features[j], self.features[j+1], kernel_size=3, stride=2, padding=1,
                                           output_padding=1))
                    self.layers_decoder.append(nn.ReLU())

        self.encoder_()
        self.decoder_()

    def encoder_(self):
        self.encoder = nn.Sequential(*self.layers_encoder)

    def decoder_(self):
        self.decoder = nn.Sequential(*self.layers_decoder)

    def forward(self, mzg):
        encoded = self.encoder(mzg)
        decoded = self.decoder(encoded)
        return decoded

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    @staticmethod
    def reversList(layers_list):
        layers_list.reverse()
        return layers_list

    def training_step(self, train_batch, batch_idx):
        input_image, target_image = train_batch
        reconstructed_x = self.forward(input_image)

        loss = self.criterion(reconstructed_x, target_image)

        return {'loss': loss}


    def validation_step(self, val_batch, batch_idx):
        input_image, target_image = val_batch
        reconstructed_x = self.forward(input_image)

        loss = self.criterion(reconstructed_x, target_image)

        return {'loss': loss}

    def test_step(self, test_batch, batch_idx):
        input_image, target_image = test_batch
        reconstructed_x = self.forward(input_image)

        # Save prediction result
        save_image(reconstructed_x, f'content/test_prediction_result/img{self.number}.png')
        self.number = self.number+1

        loss = self.criterion(reconstructed_x, target_image)

        return {'loss': loss}

    def training_epoch_end(self, outputs):
        pass # it can be add into this function whatever we want

    def validation_epoch_end(self, outputs):
        pass # it can be add into this function whatever we want