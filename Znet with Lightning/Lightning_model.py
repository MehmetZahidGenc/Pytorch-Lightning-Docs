import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from Znet_parts import DoubleConv, MaxPoolLayer, Zpoint, BatchNormalization, classifierPart
from torchmetrics import Accuracy, Recall, Precision, F1Score
from tabulate import tabulate

"""
Model structure Znet is designed by Mehmet Zahid Genç - https://github.com/MehmetZahidGenc/Znet
"""


class Znet(pl.LightningModule):
    def __init__(self, n_channels=3, n_classes=2):
        super(Znet, self).__init__()

        # Metrics
        self.accuracy = Accuracy(task='multiclass', average='micro', num_classes=n_classes)
        self.precision_ = Precision(task='multiclass', average='macro', num_classes=n_classes)
        self.recall = Recall(task='multiclass', average='macro', num_classes=n_classes)
        self.f1 = F1Score(task='multiclass', average='macro', num_classes=n_classes)

        self.n_channels = n_channels
        self.n_classes = n_classes

        self.epoch = 0

        self.maxP = MaxPoolLayer()

        self.batchNorm = BatchNormalization(num_of_features=640)

        self.block_1 = DoubleConv(in_channels=self.n_channels, mid_channels=32, out_channels=64)
        self.block_2 = DoubleConv(in_channels=64, mid_channels=64, out_channels=128)
        self.block_3 = DoubleConv(in_channels=128, mid_channels=128, out_channels=128)
        self.Zpoint_1 = Zpoint()
        self.block_4 = DoubleConv(in_channels=131, mid_channels=195, out_channels=256)
        self.block_5 = DoubleConv(in_channels=256, mid_channels=512, out_channels=512)
        self.Zpoint_2 = Zpoint()

        self.classifier_part = classifierPart(n_classes=self.n_classes)

    def forward(self, x):
        identify_1 = x

        x1 = self.block_1(x)
        x1 = self.maxP(x1)

        x2 = self.block_2(x1)
        x2 = self.maxP(x2)

        identify_2 = x2

        x3 = self.block_3(x2)

        x4 = self.Zpoint_1(x3, identify_1, is_MaxPool=True)

        x5 = self.block_4(x4)
        x5 = self.maxP(x5)

        x6 = self.block_5(x5)

        x7 = self.Zpoint_2(x6, identify_2, is_MaxPool=True)

        bn = self.batchNorm(x7)

        x8 = self.maxP(bn)

        flatten_x = torch.flatten(x8, 1)

        output = self.classifier_part(flatten_x)

        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    @staticmethod
    def cross_entropy_loss(logits, labels):
        return F.cross_entropy(logits, labels)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)

        loss = self.cross_entropy_loss(logits, y)
        accuracy = self.accuracy(logits, y)
        precision = self.precision_(logits, y)
        recall = self.recall(logits, y)
        f1score = self.f1(logits, y)

        self.log_dict(
            {'train_loss': loss, 'train_accuracy': accuracy, 'train_precision': precision, 'train_recall': recall,
             'train_f1_score': f1score},
            on_step=False, on_epoch=True, prog_bar=True)

        return {'loss': loss, "scores": logits, "y": y}

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)

        loss = self.cross_entropy_loss(logits, y)
        accuracy = self.accuracy(logits, y)
        precision = self.precision_(logits, y)
        recall = self.recall(logits, y)
        f1score = self.f1(logits, y)

        self.log_dict(
            {'val_loss': loss, 'val_accuracy': accuracy, 'val_precision': precision, 'val_recall': recall,
             'val_f1_score': f1score},
            on_step=False, on_epoch=True, prog_bar=True)

        return {'loss': loss, "scores": logits, "y": y}

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        logits = self.forward(x)

        loss = self.cross_entropy_loss(logits, y)
        accuracy = self.accuracy(logits, y)
        precision = self.precision_(logits, y)
        recall = self.recall(logits, y)
        f1score = self.f1(logits, y)

        self.log_dict(
            {'test_loss': loss, 'test_accuracy': accuracy, 'test_precision': precision, 'test_recall': recall,
             'test_f1_score': f1score},
            on_step=False, on_epoch=True, prog_bar=True)

        return {'loss': loss, "scores": logits, "y": y}

    def training_epoch_end(self, outputs):
        Tacc = 0
        Tprec = 0
        Trec = 0
        Tf1_score = 0

        for pred in outputs:
            acc = float(self.accuracy(pred['scores'], pred['y']))
            prec = float(self.precision_(pred['scores'], pred['y']))
            rec = float(self.recall(pred['scores'], pred['y']))
            f1_score = float(self.f1(pred['scores'], pred['y']))

            Tacc = Tacc+acc
            Tprec = Tprec+prec
            Trec = Trec+rec
            Tf1_score = Tf1_score+f1_score

        data = [[f"Train[Epoch: {self.epoch}]", Tacc / len(outputs), Tprec / len(outputs), Trec / len(outputs),
                 Tf1_score / len(outputs)]]

        headers = ["Type", 'Accuracy', 'Precision', 'Recall', 'F1 Score']

        self.epoch += 1
        print(tabulate(data, headers=headers))
        print("\n")
        print("\n")

    def validation_epoch_end(self, outputs):
        Vacc = 0
        Vprec = 0
        Vrec = 0
        Vf1_score = 0

        for pred in outputs:
            acc = float(self.accuracy(pred['scores'], pred['y']))
            prec = float(self.precision_(pred['scores'], pred['y']))
            rec = float(self.recall(pred['scores'], pred['y']))
            f1_score = float(self.f1(pred['scores'], pred['y']))

            Vacc = Vacc+acc
            Vprec = Vprec+prec
            Vrec = Vrec+rec
            Vf1_score = Vf1_score+f1_score

        data = [[f"Validation[Epoch: {self.epoch}]", Vacc / len(outputs), Vprec / len(outputs), Vrec / len(outputs),
                 Vf1_score / len(outputs)]]

        headers = ["Type", 'Accuracy', 'Precision', 'Recall', 'F1 Score']

        print(tabulate(data, headers=headers))
        print(' ')