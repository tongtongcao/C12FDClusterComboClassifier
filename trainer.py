import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


class ClusterComboMLP(pl.LightningModule):
    def __init__(
        self,
        input_dim=6,
        hidden_dim=16,
        num_layers=2,
        lr=1e-3,
        pos_weight=None,
        dynamic_epochs=5,
        ema_alpha=0.9,
    ):
        """
        Multi-layer perceptron (MLP) for cluster-based binary classification with optional
        dynamic positive sample weighting.

        :param input_dim: Number of input features
        :param hidden_dim: Number of neurons in each hidden layer
        :param num_layers: Number of hidden layers
        :param lr: Learning rate
        :param pos_weight: Global weight for positive samples (float)
        :param dynamic_epochs: Number of initial epochs to use batch-wise dynamic weighting
        :param ema_alpha: Smoothing factor for exponential moving average of pos_weight
        """
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.pos_weight_global = pos_weight
        self.dynamic_epochs = dynamic_epochs
        self.ema_alpha = ema_alpha
        self.pos_weight_smooth = pos_weight  # Current smoothed weight

        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)

        self.loss_fn = nn.BCELoss(reduction='none')

    def forward(self, x):
        """
        Forward pass through the MLP.

        :param x: Input tensor of shape (batch_size, input_dim)
        :return: Predicted probabilities of shape (batch_size,)
        """
        return self.model(x).squeeze(-1)

    def _compute_dynamic_pos_weight(self, y):
        """
        Compute dynamic positive sample weight for the current batch.

        :param y: Ground truth labels (0 or 1)
        :return: Smoothed positive weight for loss computation
        """
        num_pos = (y == 1).sum().item()
        num_neg = (y == 0).sum().item()
        if num_pos == 0:
            return self.pos_weight_smooth  # Avoid division by zero
        batch_weight = num_neg / max(num_pos, 1)
        # Smooth update using exponential moving average
        self.pos_weight_smooth = (
            self.ema_alpha * self.pos_weight_smooth + (1 - self.ema_alpha) * batch_weight
        )
        return self.pos_weight_smooth

    def _compute_loss(self, y_hat, y):
        """
        Compute the weighted binary cross-entropy loss.

        :param y_hat: Predicted probabilities
        :param y: Ground truth labels
        :return: Scalar loss
        """
        loss = self.loss_fn(y_hat, y)

        if self.pos_weight_global is not None:
            current_epoch = getattr(self.trainer, "current_epoch", 0)
            if current_epoch < self.dynamic_epochs:
                # Phase 1: Use batch-wise dynamic weight
                pos_weight = self._compute_dynamic_pos_weight(y)
            else:
                # Phase 2: Use smoothed weight
                pos_weight = self.pos_weight_smooth
            weights = torch.ones_like(y)
            weights[y == 1] = pos_weight
            loss = loss * weights
        return loss.mean()

    def training_step(self, batch, batch_idx):
        """
        Training step for PyTorch Lightning.

        :param batch: Tuple of (inputs, labels)
        :param batch_idx: Batch index
        :return: Loss tensor
        """
        x, y = batch
        y_hat = self(x)
        loss = self._compute_loss(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("pos_weight", self.pos_weight_smooth, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for PyTorch Lightning.

        :param batch: Tuple of (inputs, labels)
        :param batch_idx: Batch index
        """
        x, y = batch
        y_hat = self(x)
        loss = self._compute_loss(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        """
        Configure optimizer and learning rate scheduler.

        :return: Dictionary with optimizer and lr_scheduler
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


class LossTracker(Callback):
    """
    Callback to track training and validation losses across epochs.
    """
    def __init__(self):
        self.train_losses = []
        self.val_losses = []

    def on_train_epoch_end(self, trainer, pl_module):
        """
        Called at the end of each training epoch.

        :param trainer: Trainer object
        :param pl_module: LightningModule being trained
        """
        self.train_losses.append(trainer.callback_metrics["train_loss"].item())

    def on_validation_epoch_end(self, trainer, pl_module):
        """
        Called at the end of each validation epoch.

        :param trainer: Trainer object
        :param pl_module: LightningModule being validated
        """
        self.val_losses.append(trainer.callback_metrics["val_loss"].item())
