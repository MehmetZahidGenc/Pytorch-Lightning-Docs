from model import ConvGAN
from custom_data import CustomData
import pytorch_lightning as pl
from torch.utils.data import DataLoader

model_convGAN = ConvGAN(in_channel=3, num_of_z=100, generator_dim=64, discriminator_dim=64, batch_size=32)

trainer = pl.Trainer(max_epochs=10)

train_dataset = CustomData(data_dir='dataset/train', image_size=64, num_channel=3)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = CustomData(data_dir='dataset/val', image_size=64, num_channel=3)
val_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=False)

trainer.fit(model_convGAN,  train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)