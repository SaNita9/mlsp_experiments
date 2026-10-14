import torch
import os
from timm import create_model
from torch.utils.data import Dataset
import lightning as L
import torchvision.io as io
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

class Classifier(L.LightningModule):
    def __init__(self, base_network, feature_dim, 
                 num_classes, weight_decay, 
                 betas, start_lr, end_lr, 
                 eps, warmup_epochs=2, 
                 backbone_lr_mult=1.0, freeze_backbone=False,
                 max_epochs=50, label_smoothing=0.0):
        
        super().__init__()
        self.save_hyperparameters(ignore=["base_network"])
        feature_dim = feature_dim or getattr(base_network, "num_features", None)
        self.backbone = base_network
        self.head = nn.Linear(feature_dim, num_classes)

        self.test_per_class_acc = torchmetrics.Accuracy(
            task="multiclass", 
            num_classes=num_classes, 
            average="none"
        )

        self.train_acc = torchmetrics.Accuracy("multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy("multiclass", num_classes=num_classes)
        self.test_acc = torchmetrics.Accuracy("multiclass", num_classes=num_classes)

        
        
        self.test_conf_matrix = torchmetrics.ConfusionMatrix(
            task="multiclass", num_classes=num_classes
        )

        self.test_f1 = torchmetrics.F1Score(
                    task="multiclass", num_classes=num_classes
        )
        self.test_kappa = torchmetrics.CohenKappa(
                    task="multiclass", num_classes=num_classes
        )

        self.test_qwk = torchmetrics.CohenKappa(
                            task="multiclass", num_classes=num_classes,
                            weights="quadratic"
                )

        self.train_loss_history = []
        self.val_loss_history = []

        if self.hparams.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False


    def forward(self, x):
        return self.head(self.backbone(x))

    def train(self, mode=True):
        # Override so a frozen backbone stays in eval mode,
        # otherwise BatchNorm running stats keep updating even with
        # requires_grad=False.
        super().train(mode)
        if self.hparams.freeze_backbone:
            self.backbone.eval()
        return self

    @staticmethod
    def _split_params(module):
        """Return (weights, others) for the trainable params of a module."""
        weights, others = [], []
        for param in module.parameters():
            if not param.requires_grad:   # skip frozen params
                continue
            if param.ndim > 1:
                weights.append(param)
            else:
                others.append(param)
        return weights, others

    def configure_optimizers(self):
        hp = self.hparams

        backbone_lr = hp.start_lr * hp.backbone_lr_mult
        head_lr = hp.start_lr

        #   weights (2-D+ tensors: conv kernels, linear matrices) -> get weight decay
        #   others  (1-D tensors: biases, BatchNorm scale/shift)   -> no weight decay
        backbone_weights, backbone_others = self._split_params(self.backbone)
        head_weights, head_others = self._split_params(self.head)

        param_groups = []

        if len(backbone_weights) > 0:
            param_groups.append({"params": backbone_weights, "lr": backbone_lr, "weight_decay": hp.weight_decay})
        if len(backbone_others) > 0:
            param_groups.append({"params": backbone_others, "lr": backbone_lr, "weight_decay": 0.0})

        param_groups.append({"params": head_weights, "lr": head_lr, "weight_decay": hp.weight_decay})
        param_groups.append({"params": head_others,  "lr": head_lr, "weight_decay": 0.0})

        optimizer = optim.AdamW(
                param_groups,
                betas=hp.betas, 
                eps=hp.eps,
                weight_decay=hp.weight_decay
            )
        warmup = optim.lr_scheduler.LinearLR(
                optimizer, 
                start_factor=0.01, 
                total_iters=hp.warmup_epochs)
        
        cosine = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=max(hp.max_epochs - hp.warmup_epochs, 1),
            eta_min=hp.end_lr)
        
        lr_scheduler = optim.lr_scheduler.SequentialLR(
                        optimizer, 
                        schedulers=[warmup, cosine], milestones=[hp.warmup_epochs])

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": lr_scheduler, "interval": "epoch"},
        }
        
    def _calculate_loss(self, batch, mode='train'):
        feats, labels = batch
        preds = self(feats)
        loss = F.cross_entropy(preds, labels, label_smoothing=self.hparams.label_smoothing)
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        self.log(mode + '_loss', loss, on_epoch=True, prog_bar=False, batch_size=len(labels))
        self.log(mode + '_acc', acc, on_epoch=True, prog_bar=False, batch_size=len(labels))
        return loss       
        
    def training_step(self, batch, batch_idx):
        return self._calculate_loss(batch, mode='train')

    def on_train_epoch_end(self):
        avg_loss = self.trainer.callback_metrics.get('train_loss')
        if avg_loss is not None:
            self.train_loss_history.append(avg_loss.item())

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode='val')

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            return
            
        metrics = self.trainer.callback_metrics
        
        t_loss = metrics.get('train_loss', torch.tensor(0.0)).item()
        t_acc  = metrics.get('train_acc', torch.tensor(0.0)).item()
        v_loss = metrics.get('val_loss', torch.tensor(0.0)).item()
        v_acc  = metrics.get('val_acc', torch.tensor(0.0)).item()
        
        epoch = self.current_epoch
        
        if epoch == 0:
            print(f"\n{'Epoch':^7} | {'Train Loss':^12} | {'Train Acc':^11} | {'Val Loss':^10} | {'Val Acc':^9}")
            print("-" * 61)
            
        print(f"{epoch:^7} | {t_loss:^12.4f} | {t_acc:^11.4f} | {v_loss:^10.4f} | {v_acc:^9.4f}")

        avg_val_loss = self.trainer.callback_metrics.get('val_loss')
        
        if avg_val_loss is not None:
            self.val_loss_history.append(avg_val_loss.item())
        
    def test_step(self, batch, batch_idx):
        feats, labels = batch
        preds = self(feats)
        self._calculate_loss(batch, mode='test')
        self.test_per_class_acc(preds, labels)
        self.test_conf_matrix(preds, labels)
        self.test_f1(preds, labels)
        self.test_kappa(preds, labels)
        self.test_qwk(preds, labels)

    def on_test_epoch_end(self):
        per_class_accuracies = self.test_per_class_acc.compute()
        
        for class_idx, acc in enumerate(per_class_accuracies):
            print(f"Class {class_idx} Accuracy: {acc.item():.4f} ({acc.item() * 100:.2f}%)")
            # self.log(f"test_acc_class_{class_idx}", acc, prog_bar=True)
        f1_score = self.test_f1.compute()
        print(f"Overall F1 Score:   {f1_score.item():.4f}\n")
        kappa = self.test_kappa.compute()
        print(f"Overall kappa Score:   {kappa.item():.4f}\n")

        qwk = self.test_qwk.compute()
        print(f"Quadratic weighted kappa Score:   {qwk.item():.4f}\n")

        conf_matrix = self.test_conf_matrix.compute().cpu().numpy()

        disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=[0, 1, 2, 3, 4, 5])
        disp.plot(cmap='Blues', values_format='d')
        plt.title("Test Confusion Matrix")
        
        
        # plt.tight_layout()
        # plt.savefig("confusion_matrix.png", dpi=300)
        
        plt.show() 
        
        plt.close()
        
        # Reset metrics
        self.test_per_class_acc.reset()
        self.test_conf_matrix.reset()
        self.test_f1.reset()
        self.test_kappa.reset()
        self.test_qwk.reset()




# class ConvModel(L.LightningModule):
#     def __init__(self, model_name: str = "resnet18", pretrained: bool = True, num_classes: int = 6):
#             super().__init__()
#             self.save_hyperparameters()

#             self.model = create_model(
#                 model_name, 
#                 pretrained=pretrained,
#                 num_classes=num_classes
#             )
#             self.num_features = self.model.num_features

#     def forward(self, x):
#         return self.model(x)


# class LogisticRegression(L.LightningModule):
    
#     def __init__(self, feature_dim, num_classes, lr, weight_decay, max_epochs=100):
#         super().__init__()
#         self.save_hyperparameters()

#         self.model = nn.Linear(feature_dim, num_classes)

#     def configure_optimizers(self):
#         optimizer = optim.AdamW(self.parameters(), 
#                                 lr=self.hparams.lr, 
#                                 weight_decay=self.hparams.weight_decay)
#         lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
#                                                       milestones=[int(self.hparams.max_epochs*0.6), 
#                                                                   int(self.hparams.max_epochs*0.8)], 
#                                                       gamma=0.1)
#         return [optimizer], [lr_scheduler]
        
#     def _calculate_loss(self, batch, mode='train'):
#         feats, labels = batch
#         preds = self.model(feats)
#         loss = F.cross_entropy(preds, labels, label_smoothing=self.hparams.label_smoothing)
#         acc = (preds.argmax(dim=-1) == labels).float().mean()

#         self.log(mode + '_loss', loss)
#         self.log(mode + '_acc', acc)
#         return loss        
        
#     def training_step(self, batch, batch_idx):
#         return self._calculate_loss(batch, mode='train')
        
#     def validation_step(self, batch, batch_idx):
#         self._calculate_loss(batch, mode='val')
        
#     def test_step(self, batch, batch_idx):
#         self._calculate_loss(batch, mode='test')

# class Pretrained_Model(L.LightningModule):
#     def __init__(self, backbone, feature_dim, num_classes, lr, weight_decay, max_epochs=100):
#         super().__init__()
#         self.save_hyperparameters()

        
#         self.model = nn.Sequential(
#             backbone,
#             nn.Linear(feature_dim, num_classes)
#             )

#     def configure_optimizers(self):
#         optimizer = optim.AdamW(self.parameters(), 
#                                 lr=self.hparams.lr, 
#                                 weight_decay=self.hparams.weight_decay)
#         lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
#                                                       milestones=[int(self.hparams.max_epochs*0.6), 
#                                                                   int(self.hparams.max_epochs*0.8)], 
#                                                       gamma=0.1)
#         return [optimizer], [lr_scheduler]
        
#     def _calculate_loss(self, batch, mode='train'):
#         feats, labels = batch
#         preds = self.model(feats)
#         loss = F.cross_entropy(preds, labels)
#         acc = (preds.argmax(dim=-1) == labels).float().mean()

#         self.log(mode + '_loss', loss)
#         self.log(mode + '_acc', acc)
#         return loss        
        
#     def training_step(self, batch, batch_idx):
#         return self._calculate_loss(batch, mode='train')
        
#     def validation_step(self, batch, batch_idx):
#         self._calculate_loss(batch, mode='val')
        
#     def test_step(self, batch, batch_idx):
#         self._calculate_loss(batch, mode='test')

