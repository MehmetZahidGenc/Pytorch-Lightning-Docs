from varAE_model import VAE
from custom_data import CustomData
import pytorch_lightning as pl
from torch.utils.data import DataLoader

model_vae = VAE(in_channel=3, image_size=128, hidden_dim=200, z_dim=20)

"""
Dataset
    - Train
        - Input
            - Image1
            - Image2
        - Target
            - Image11
            - Image22
    - Val
    - Test(optional)
"""

train_dataset = CustomData(target_root_dir='data/train/target', input_root_dir='data/train/input')
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = CustomData(target_root_dir='data/val/target', input_root_dir='data/val/target')
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=10)

trainer.fit(model_vae,  train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)