from Lightning_model import Znet
from Lightning_data import CustomData
import pytorch_lightning as pl

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

      --> classes.txt
      --> test(optional)
            same architecture as train
"""

Znet_model = Znet(n_channels=3, n_classes=5)

data_module = CustomData(data_dir="/content/dataset/", train_batch_size=16, val_batch_size=16, test_data=False)

trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=5)

trainer.fit(Znet_model, data_module)