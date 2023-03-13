from model import ConvAE
from custom_data import CustomData
import pytorch_lightning as pl
from torch.utils.data import DataLoader

model = ConvAE(in_channel=3, features=[16, 32, 64])

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

trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=5)

trainer.fit(model,  train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)