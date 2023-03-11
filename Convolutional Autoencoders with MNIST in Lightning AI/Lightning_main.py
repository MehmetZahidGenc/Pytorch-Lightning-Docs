from Lightning_model import ConvAE
from Lightning_data import CustomData
import pytorch_lightning as pl

model = ConvAE(in_channel=1, features=[16, 32, 64])

data_module = CustomData(data_dir="/content/dataset/", train_batch_size=16, val_batch_size=16, test_data=False)

trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=5)

trainer.fit(model, data_module)