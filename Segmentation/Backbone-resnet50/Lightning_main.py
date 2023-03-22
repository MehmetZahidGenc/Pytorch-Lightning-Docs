import pytorch_lightning as pl
from Lightning_model import SegmentationModel
from Lightning_data import SegmentationDataModule

"""
Let's train the model with our custom dataset
"""

# create model
model = SegmentationModel(input_channels=3, output_channels=1, learning_rate=1e-3)

# create lightning datamodule
# please use data which include 3 channel input and 3 or 1 channel output.
data_module = SegmentationDataModule(input_channels=3,
                                     output_channels=1,
                                     train_mask_root_dir='Dataset_Unet/train_masks',
                                     train_input_root_dir='Dataset_Unet/train_images',
                                     train_batch_size=16,
                                     val_mask_root_dir='Dataset_Unet/val_masks',
                                     val_input_root_dir='Dataset_Unet/val_images',
                                     val_batch_size=16)

# Trainer
trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=1000, log_every_n_steps=2)

# fitting
trainer.fit(model, data_module)