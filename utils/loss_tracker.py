import matplotlib.pyplot as plt
import pytorch_lightning as pl
from IPython.display import display, HTML
import pandas as pd


class LossTracker(pl.Callback):
    def __init__(self):
        self.train_loss = []
        self.train_acc= []
        self.val_loss = []
        self.val_acc = []

    def on_train_epoch_end(self, trainer, pl_module):
        loss = trainer.callback_metrics.get("train_loss").cpu()
        acc = trainer.callback_metrics.get("train_acc").cpu()
        self.train_loss.append(loss)
        self.train_acc.append(acc)

    def on_validation_epoch_end(self, trainer, pl_module):
        loss = trainer.callback_metrics.get("val_loss").cpu()
        loss = loss.cpu()
        acc = trainer.callback_metrics.get("val_acc").cpu()
        self.val_loss.append(loss)
        self.val_acc.append(acc)

        #if len(self.train_loss):
        metrics = {
            "Epoch": [trainer.current_epoch],
            "train_loss": [self.train_loss[-1].item() if len(self.train_loss) else -1],
            "train_acc": [self.train_acc[-1].item() if len(self.train_acc) else -1],
            "val_loss": [self.val_loss[-1].item()],
            "val_acc": [self.val_acc[-1].item()],
        }
        metrics = pd.DataFrame(metrics)

        # Custom CSS styling
        pl_style = """
        <style>
            .pl-table {
                border-collapse: collapse;
                margin: 10px 0;
                font-family: monospace;
                font-size: 14px;
                width: auto;
            }
            .pl-table th {
                border-bottom: 2px solid #000;
                font-weight: bold;
                padding: 4px 8px;
                text-align: left;
            }
            .pl-table td {
                padding: 4px 8px;
                text-align: left;
            }
            .pl-table tr:nth-child(even) {
                background-color: #f8f8f8;
            }
        </style>
        """
        
        # Display the table.
        display(HTML(pl_style + metrics.to_html(classes='pl-table', index=False, float_format='{:,.6f}'.format)))

    def plot(self):
        # Clear ouput.
        #display.clear_output(wait=True)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot loss
        axes[0].plot(self.train_loss, label="Train loss")
        axes[0].plot(self.val_loss[1:], label="Validation loss")
        axes[0].set_xlabel("Epochs")
        axes[0].set_ylabel("Loss")

        # Plot accuracy
        axes[1].plot(self.train_acc, label="Train accuracy", color="purple")
        axes[1].plot(self.val_acc[1:], label="Validation accuracy", color="green")
        axes[1].set_xlabel("Epochs")
        axes[1].set_ylabel("Accuracy")

        for ax in axes:
            ax.legend()
            ax.grid(True)
        
        plt.tight_layout()
        plt.show()
        plt.close()