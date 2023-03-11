"""
Basic LightningModule structure must be as

class LightningNet(pl.LightningModule):
    def __init__(...):
        ...

    def forward(...):
        ...

    def training_step(...):
        ...

    def configure_optimizers(...):
        ...

    def training_epoch_end(...):
        ...

    def validation_epoch_end(...):
        ...

"""


"""
METHOD-1 ==> This method calculate metrics value of only current train epoch
"""

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



"""
METHOD-2 ==> This method calculate average of metrics value of passed train epochs
"""

def training_epoch_end(self, outputs):
    data = [[f"Train[Epoch: {self.epoch}]", self.accuracy.compute(), self.precision_.compute(), self.recall.compute(),
             self.f1.compute()]]

    headers = ["Type", 'Accuracy', 'Precision', 'Recall', 'F1 Score']

    self.epoch += 1
    print(tabulate(data, headers=headers))
    print("\n")