import matplotlib.pyplot as plt
import pytorch_lightning as pl
from IPython import display

class LossTracker(pl.Callback):
    def __init__(self):
        self.train_loss = []
        self.val_loss = []
        self.val_acc = []

    #def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
    def on_train_epoch_end(self, trainer, pl_module):
        loss = trainer.callback_metrics.get("train_loss").cpu()
        self.train_loss.append(loss)
        #self.plot()

    #def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
    def on_validation_epoch_end(self, trainer, pl_module):
        loss = trainer.callback_metrics.get("val_loss")
        if loss is None:
            return
        loss = loss.cpu()
        acc = trainer.callback_metrics.get("val_acc").cpu()
        self.val_loss.append(loss)
        self.val_acc.append(acc)

    def plot(self):
        # Clear ouput.
        #display.clear_output(wait=True)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot loss
        axes[0].plot(self.train_loss, label="Train loss")
        axes[0].plot(self.val_loss[1:], label="Validation loss")
        axes[0].set_xlabel("Iterations")
        axes[0].set_ylabel("Loss")

        # Plot accuracy
        axes[1].plot(self.val_acc[1:], label="Validation accuracy", color="green")
        axes[1].set_xlabel("Iterations")
        axes[1].set_ylabel("Accuracy")

        for ax in axes:
            ax.legend()
            ax.grid(True)
        
        plt.tight_layout()
        plt.show()
        plt.close()