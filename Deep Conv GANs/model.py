import pytorch_lightning as pl
from torchvision.utils import save_image
import torch
import torchvision
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

"""
               Deep Convolutional GANs with Lightning AI
                         _____________________
        Real Image -->   |  Deep Conv GANs   |  --> Fake Image 
                         |___________________|
"""


# custom weights initialization called on netG and netD

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


""" Discriminator Class """


def discriminator_conv(in_channel, out_channel, kernel_size=3, stride=2, padding=1, batch_norm=True, bias=False):
    layers = [nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding, bias=bias)]

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channel))

    return nn.Sequential(*layers)


class Discriminator(nn.Module):
    def __init__(self, number_of_channel, discriminator_dim):
        super(Discriminator, self).__init__()

        self.disc_conv1 = discriminator_conv(number_of_channel, discriminator_dim, kernel_size=4, stride=2, padding=1,
                                             batch_norm=False)
        self.disc_conv2 = discriminator_conv(discriminator_dim, discriminator_dim * 2, kernel_size=4, stride=2,
                                             padding=1, batch_norm=True)
        self.disc_conv3 = discriminator_conv(discriminator_dim * 2, discriminator_dim * 4, kernel_size=4, stride=2,
                                             padding=1, batch_norm=True)
        self.disc_conv4 = discriminator_conv(discriminator_dim * 4, discriminator_dim * 8, kernel_size=4, stride=2,
                                             padding=1, batch_norm=True)

        self.disc_conv5 = discriminator_conv(discriminator_dim * 8, 1, kernel_size=4, stride=1, padding=0,
                                             batch_norm=False)

    def forward(self, mzg):
        out = F.leaky_relu(self.disc_conv1(mzg), inplace=True, negative_slope=0.2)
        out = F.leaky_relu(self.disc_conv2(out), inplace=True, negative_slope=0.2)
        out = F.leaky_relu(self.disc_conv3(out), inplace=True, negative_slope=0.2)
        out = F.leaky_relu(self.disc_conv4(out), inplace=True, negative_slope=0.2)

        out = torch.sigmoid(self.disc_conv5(out))

        return out


""" Generator Class """


def generator_conv(in_channel, out_channel, kernel_size=3, stride=2, padding=1, batch_norm=True, bias=False):
    layers = [
        nn.ConvTranspose2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channel))

    return nn.Sequential(*layers)


class Generator(nn.Module):
    def __init__(self, num_of_z, number_of_channel, generator_dim):
        super(Generator, self).__init__()

        self.gen_conv1 = generator_conv(num_of_z, generator_dim * 8, kernel_size=4, stride=1, padding=0,
                                        batch_norm=True)
        self.gen_conv2 = generator_conv(generator_dim * 8, generator_dim * 4, kernel_size=4, stride=2, padding=1,
                                        batch_norm=True)
        self.gen_conv3 = generator_conv(generator_dim * 4, generator_dim * 2, kernel_size=4, stride=2, padding=1,
                                        batch_norm=True)
        self.gen_conv4 = generator_conv(generator_dim * 2, generator_dim, kernel_size=4, stride=2, padding=1,
                                        batch_norm=True)

        self.gen_conv5 = generator_conv(generator_dim, number_of_channel, kernel_size=4, stride=2, padding=1,
                                        batch_norm=False)

    def forward(self, mzg):
        out = F.relu(self.gen_conv1(mzg))
        out = F.relu(self.gen_conv2(out))
        out = F.relu(self.gen_conv3(out))
        out = F.relu((self.gen_conv4(out)))

        out = torch.tanh(self.gen_conv5(out))

        return out


class ConvGAN(pl.LightningModule):
    def __init__(self, in_channel, num_of_z, generator_dim, discriminator_dim, batch_size):
        super(ConvGAN, self).__init__()
        self.in_channel = in_channel
        self.num_of_z = num_of_z
        self.generator_dim = generator_dim
        self.discriminator_dim = discriminator_dim

        self.batch_size = batch_size

        self.number = 0
        self.step = 0

        self.automatic_optimization = False

        self.fixed_noise = torch.randn(32, self.num_of_z, 1, 1)

        self.writer_real = SummaryWriter(f"logs/real")
        self.writer_fake = SummaryWriter(f"logs/fake")

        self.criterion = nn.BCELoss()

        self.gen = Generator(num_of_z=self.num_of_z, number_of_channel=self.in_channel,
                             generator_dim=self.generator_dim)

        self.disc = Discriminator(number_of_channel=self.in_channel, discriminator_dim=self.discriminator_dim)

    def forward(self, real, noise):
        fake = self.gen(noise)

        ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        disc_real = self.disc(real).reshape(-1)
        disc_fake = self.disc(fake.detach()).reshape(-1)

        ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        output = self.disc(fake).reshape(-1)

        return disc_real, disc_fake, output

    def configure_optimizers(self):
        opt_gen = torch.optim.Adam(self.gen.parameters(), lr=0.0002, betas=(0.5, 0.999))
        opt_disc = torch.optim.Adam(self.disc.parameters(), lr=0.0002, betas=(0.5, 0.999))
        return opt_gen, opt_disc

    def training_step(self, train_batch, batch_idx):
        # optimizers
        opt_gen, opt_disc = self.optimizers()

        real = train_batch
        noise = torch.randn(self.batch_size, self.num_of_z, 1, 1)

        disc_real, disc_fake, output = self.forward(real, noise)

        loss_disc_real = self.criterion(disc_real, torch.ones_like(disc_real))
        loss_disc_fake = self.criterion(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = (loss_disc_real+loss_disc_fake) / 2

        loss_gen = self.criterion(output, torch.ones_like(output))

        self.log_dict(
            {'train loss disc': loss_disc, 'train loss_gen': loss_gen},
            on_step=False, on_epoch=True, prog_bar=True)

        return {'loss disc': loss_disc, 'loss_gen': loss_gen}

    def validation_step(self, val_batch, batch_idx):
        # optimizers
        opt_gen, opt_disc = self.optimizers()

        real = val_batch
        noise = torch.randn(self.batch_size, self.num_of_z, 1, 1)

        disc_real, disc_fake, output = self.forward(real, noise)

        loss_disc_real = self.criterion(disc_real, torch.ones_like(disc_real))
        loss_disc_fake = self.criterion(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = (loss_disc_real+loss_disc_fake) / 2

        loss_gen = self.criterion(output, torch.ones_like(output))

        self.log_dict(
            {'val loss disc': loss_disc, 'val loss_gen': loss_gen},
            on_step=False, on_epoch=True, prog_bar=True)

        return {'loss disc': loss_disc, 'loss_gen': loss_gen}

    def test_step(self, test_batch, batch_idx):
        # optimizers
        opt_gen, opt_disc = self.optimizers()

        real = test_batch
        noise = torch.randn(self.batch_size, self.num_of_z, 1, 1)

        disc_real, disc_fake, output = self.forward(real, noise)

        # Print losses occasionally and print to tensorboard
        if batch_idx % 10 == 0:
            with torch.no_grad():
                fake = self.gen(self.fixed_noise)
                # take out (up to) 32 examples
                img_grid_real = torchvision.utils.make_grid(
                    real[:32], normalize=True
                )
                img_grid_fake = torchvision.utils.make_grid(
                    fake[:32], normalize=True
                )

                self.writer_real.add_image("Real", img_grid_real, global_step=self.step)
                self.writer_fake.add_image("Fake", img_grid_fake, global_step=self.step)

            self.step += 1

        loss_disc_real = self.criterion(disc_real, torch.ones_like(disc_real))
        loss_disc_fake = self.criterion(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = (loss_disc_real+loss_disc_fake) / 2

        loss_gen = self.criterion(output, torch.ones_like(output))

        return {'loss disc': loss_disc, 'loss_gen': loss_gen}