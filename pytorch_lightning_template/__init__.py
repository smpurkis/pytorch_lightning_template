import pytorch_lightning as pl
import torch
import torchinfo as ti


class LightningModuleTemplate(pl.LightningModule):
    def __init__(self, input_shape=None, loss_fn=None, opt_fn=torch.optim.Adam, acc_fn=None, lr=1e-4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch = 0
        self.input_shape = input_shape
        self.acc_fn = acc_fn
        self.loss_fn = loss_fn
        self.opt_fn = opt_fn
        self.lr = lr
        self.metrics = {}

    def configure_optimizers(self):
        optimizer = self.opt_fn(self.parameters(), lr=self.lr)
        return optimizer

    def step(self, batch):
        x = batch[0]
        y = batch[1]
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y)
        step_outputs = {
            "loss": loss
        }
        self.log("step_loss", loss, prog_bar=True, logger=True)
        if self.acc_fn:
            acc = self.acc_fn(y_hat, y)
            step_outputs["acc"] = acc
            self.log("step_acc", acc, prog_bar=True, logger=True)
        return step_outputs

    def training_step(self, batch, batch_idx):
        step_outputs = self.step(batch)
        return step_outputs

    def validation_step(self, batch, batch_idx):
        step_outputs = self.step(batch)
        return step_outputs

    def format_metrics(self, epoch_outputs):
        epoch_metrics = {}
        loss = torch.mean(torch.tensor([e["loss"] for e in epoch_outputs]))
        epoch_metrics["loss"] = loss
        self.log("loss", loss, prog_bar=True, logger=True)
        if self.acc_fn:
            acc = torch.mean(torch.tensor([e["acc"] for e in epoch_outputs]))
            epoch_metrics["acc"] = acc
            self.log("acc", acc, prog_bar=True, logger=True)
        return epoch_metrics

    def training_epoch_end(self, training_epoch_outputs):
        epoch_metrics = self.format_metrics(training_epoch_outputs)
        # for metric, value in epoch_metrics.items():
        #     self.log(f"train_epoch_{metric}", value, prog_bar=True, logger=True, on_epoch=True)
        if self.val_dataloader is None:
            print(f"\nEpoch: {self.epoch}", end=" ")
            print(f"Training, loss: {epoch_metrics['loss']:5.3f}, acc: {epoch_metrics['acc']:5.3f}", end=" ")
            self.epoch += 1
        else:
            self.metrics = {"train_metrics": epoch_metrics}

    def validation_epoch_end(self, validation_epoch_outputs):
        epoch_metrics = self.format_metrics(validation_epoch_outputs)
        # for metric, value in epoch_metrics.items():
        #     self.log(f"val_epoch_{metric}", value, prog_bar=True, logger=True, on_epoch=True)
        self.metrics["val_metrics"] = epoch_metrics
        if self.metrics.get("train_metrics"):
            metric_str = f"\nEpoch: {self.epoch}  " \
                         f"Training, loss: {self.metrics['train_metrics']['loss']:5.3f}, acc: {self.metrics['train_metrics']['acc']:5.3f}  " \
                         f"Validation, loss: {epoch_metrics['loss']:5.3f}, acc: {epoch_metrics['acc']:5.3f}"
            print(metric_str, end="")
            self.epoch += 1

    def summary(self):
        if self.input_shape is None:
            raise Exception("Please set 'self.input_shape'")
        else:
            print(ti.summary(self, (1, *self.input_shape)))
