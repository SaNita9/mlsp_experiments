import torch
import os
from timm import create_model
from torch.utils.data import Dataset
import lightning as L
import torchvision.io as io
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Model(L.LightningModule):
	def __init__(self, model, feature_dim, 
	             num_classes, weight_decay, 
				 betas, start_lr, end_lr, 
				 eps, max_epochs=50):
		
		super().__init__()
		self.save_hyperparameters()
		self.model = model
		self.weight_decay  = weight_decay
		self.betas = betas
		self.max_epochs = max_epochs
		self.start_lr = start_lr
		self.end_lr = end_lr
		self.eps = eps


	def configure_optimizers(self):
		optimizer = torch.optim.AdamW(
				self.model.parameters(),
				lr=self.start_lr,
				betas=self.betas,
				weight_decay=self.weight_decay,
				eps=self.eps
			)
		lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epochs, eta_min=self.end_lr)
		return [optimizer], [lr_scheduler]
		
	def _calculate_loss(self, batch, mode='train'):
		feats, labels = batch
		preds = self.model(feats)
		loss = F.cross_entropy(preds, labels)
		acc = (preds.argmax(dim=-1) == labels).float().mean()

		self.log(mode + '_loss', loss)
		self.log(mode + '_acc', acc)
		return loss        
		
	def training_step(self, batch, batch_idx):
		return self._calculate_loss(batch, mode='train')
		
	def validation_step(self, batch, batch_idx):
		self._calculate_loss(batch, mode='val')
		
	def test_step(self, batch, batch_idx):
		self._calculate_loss(batch, mode='test')




class ConvModel(L.LightningModule):
	def __init__(self, model_name: str = "resnet18", pretrained: bool = True, num_classes: int = 6):
			super().__init__()
			self.save_hyperparameters()

			self.model = create_model(
				model_name, 
				pretrained=pretrained,
				num_classes=num_classes
			)
			self.num_features = self.model.num_features

	def forward(self, x):
		return self.model(x)


class LogisticRegression(L.LightningModule):
	
	def __init__(self, feature_dim, num_classes, lr, weight_decay, max_epochs=100):
		super().__init__()
		self.save_hyperparameters()

		self.model = nn.Linear(feature_dim, num_classes)

	def configure_optimizers(self):
		optimizer = optim.AdamW(self.parameters(), 
								lr=self.hparams.lr, 
								weight_decay=self.hparams.weight_decay)
		lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
													  milestones=[int(self.hparams.max_epochs*0.6), 
																  int(self.hparams.max_epochs*0.8)], 
													  gamma=0.1)
		return [optimizer], [lr_scheduler]
		
	def _calculate_loss(self, batch, mode='train'):
		feats, labels = batch
		preds = self.model(feats)
		loss = F.cross_entropy(preds, labels)
		acc = (preds.argmax(dim=-1) == labels).float().mean()

		self.log(mode + '_loss', loss)
		self.log(mode + '_acc', acc)
		return loss        
		
	def training_step(self, batch, batch_idx):
		return self._calculate_loss(batch, mode='train')
		
	def validation_step(self, batch, batch_idx):
		self._calculate_loss(batch, mode='val')
		
	def test_step(self, batch, batch_idx):
		self._calculate_loss(batch, mode='test')

class Pretrained_Model(L.LightningModule):
	def __init__(self, backbone, feature_dim, num_classes, lr, weight_decay, max_epochs=100):
		super().__init__()
		self.save_hyperparameters()

		
		self.model = nn.Sequential(
			backbone,
			nn.Linear(feature_dim, num_classes)
			)

	def configure_optimizers(self):
		optimizer = optim.AdamW(self.parameters(), 
								lr=self.hparams.lr, 
								weight_decay=self.hparams.weight_decay)
		lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
													  milestones=[int(self.hparams.max_epochs*0.6), 
																  int(self.hparams.max_epochs*0.8)], 
													  gamma=0.1)
		return [optimizer], [lr_scheduler]
		
	def _calculate_loss(self, batch, mode='train'):
		feats, labels = batch
		preds = self.model(feats)
		loss = F.cross_entropy(preds, labels)
		acc = (preds.argmax(dim=-1) == labels).float().mean()

		self.log(mode + '_loss', loss)
		self.log(mode + '_acc', acc)
		return loss        
		
	def training_step(self, batch, batch_idx):
		return self._calculate_loss(batch, mode='train')
		
	def validation_step(self, batch, batch_idx):
		self._calculate_loss(batch, mode='val')
		
	def test_step(self, batch, batch_idx):
		self._calculate_loss(batch, mode='test')

