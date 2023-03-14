import torch
import pytorch_lightning as pl
import torch.nn as nn
from torchvision.utils import save_image

"""
         Variational Auto Encoders with Lightning AI
                         __________
        Input Image -->  |  VAE   |  --> Target Image 
                         |________|
"""

class VAE(pl.LightningModule):
    def __init__(self, in_channel, image_size, hidden_dim, z_dim):
        super().__init__()
        self.in_channel = in_channel
        self.image_size = image_size
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim

        self.loss_fn = nn.MSELoss()

        self.number = 0 # it will be used to control test prediction result saving

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channel, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Flatten(),
            nn.Linear(4096, self.hidden_dim) # 4096, it is changeable according to input sizes
        )

        self.mu = nn.Linear(self.hidden_dim, self.z_dim)
        self.sigma = nn.Linear(self.hidden_dim, self.z_dim)

        self.decoder_first_layer = nn.Linear(self.z_dim, 4096) # 4096, it is changeable according to input sizes

        self.decoder = nn.Sequential(
            nn.Unflatten(dim=1, unflattened_size=(64, 8, 8)), # 64*8*8 = 4096

            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, bias=False, padding=1),
            nn.LeakyReLU(0.1, inplace=True),

            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, bias=False, padding=1),
            nn.LeakyReLU(0.1, inplace=True),

            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, bias=False, padding=1),
            nn.LeakyReLU(0.1, inplace=True),

            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, bias=False, padding=1),
            nn.LeakyReLU(0.1, inplace=True),

            nn.ConvTranspose2d(in_channels=32, out_channels=self.in_channel, kernel_size=3, stride=2, bias=False, padding=1),

        )

    def encoder_forw(self, mzg):
        mzg = self.encoder(mzg)

        mu = self.mu(mzg)
        sigma = self.sigma(mzg)

        epsilon = torch.randn_like(sigma)
        encoded = mu+sigma * epsilon

        return encoded, mu, sigma

    def decoder_forw(self, mzg):
        mzg = self.decoder_first_layer(mzg)

        mzg = self.decoder(mzg)

        mzg = mzg[:, :, :self.image_size, :self.image_size] # avoids tensor size mismatch issue

        decoded = torch.sigmoid(mzg)

        return decoded

    def forward(self, mzg):
        encoded, mu, sigma = self.encoder_forw(mzg)

        # output = decoded
        output = self.decoder_forw(encoded)

        return output, mu, sigma

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    @staticmethod
    def reversList(layers_list):
        layers_list.reverse()
        return layers_list

    def training_step(self, train_batch, batch_idx):
        input_image, target_image = train_batch
        reconstructed_x, mu, sigma = self.forward(input_image)

        reconst_loss = self.loss_fn(reconstructed_x, target_image)
        kl_div = - torch.sum(1+torch.log(sigma.pow(2))-mu.pow(2)-sigma.pow(2))

        loss = reconst_loss+kl_div

        return {'loss': loss}


    def validation_step(self, val_batch, batch_idx):
        input_image, target_image = val_batch
        reconstructed_x, mu, sigma = self.forward(input_image)

        reconst_loss = self.loss_fn(reconstructed_x, target_image)
        kl_div = - torch.sum(1+torch.log(sigma.pow(2))-mu.pow(2)-sigma.pow(2))

        loss = reconst_loss+kl_div

        return {'loss': loss}

    def test_step(self, test_batch, batch_idx):
        input_image, target_image = test_batch
        reconstructed_x, mu, sigma = self.forward(input_image)

        reconst_loss = self.loss_fn(reconstructed_x, target_image)
        kl_div = - torch.sum(1+torch.log(sigma.pow(2))-mu.pow(2)-sigma.pow(2))

        # Save prediction result
        save_image(reconstructed_x, f'img{self.number}.png')
        self.number = self.number+1

        loss = reconst_loss+kl_div

        return {'loss': loss}


    def training_epoch_end(self, outputs):
        pass # it can be add into this function whatever we want

    def validation_epoch_end(self, outputs):
        pass # it can be add into this function whatever we want