import lightning as L
import torch
import os
import torch.nn.functional as F
from dotenv import load_dotenv
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import v2
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from timm import create_model
from lightning.pytorch.loggers import CSVLogger
from arg_utils import get_parser, load_yaml_config, merge_config_into_args
from dataset_utils import get_simclr_dataset, get_dataloader

SEED = 42
NUM_WORKERS = 4

class ContrastiveTransformations(object):
	
	def __init__(self, base_transforms, n_views=2):
		self.base_transforms = base_transforms
		self.n_views = n_views
		
	def __call__(self, x):
		return [self.base_transforms(x) for i in range(self.n_views)]


class SimCLR(L.LightningModule):
	
	def __init__(self, model_name, pretrained, hidden_dim, start_lr, end_lr, temperature, weight_decay, max_epochs=50):
		super().__init__()
		self.save_hyperparameters(ignore=["model_name", "pretrained"])
		self.convnet = self.model = create_model(
                model_name, 
                pretrained=pretrained,
                num_classes=0
        )
	   
		backbone_out_features = self.convnet.num_features
		
		self.projection_head = nn.Sequential(
			nn.Linear(backbone_out_features, backbone_out_features), 
			nn.ReLU(inplace=True),
			nn.Linear(backbone_out_features, hidden_dim)
		)

	def configure_optimizers(self):
		optimizer = optim.AdamW(self.parameters(), 
								lr=self.hparams.start_lr, 
								weight_decay=self.hparams.weight_decay)
		lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
															T_max=self.hparams.max_epochs,
															eta_min=self.hparams.end_lr)
		return [optimizer], [lr_scheduler]
		
	def info_nce_loss(self, batch, mode='train'):
		imgs, _ = batch
		imgs = torch.cat(imgs, dim=0)
		
		h = self.convnet(imgs)
		
		feats = self.projection_head(h)
		cos_sim = F.cosine_similarity(feats[:,None,:], feats[None,:,:], dim=-1)
		self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
		cos_sim.masked_fill_(self_mask, -9e15)
		pos_mask = self_mask.roll(shifts=cos_sim.shape[0]//2, dims=0)

		# InfoNCE loss
		cos_sim = cos_sim / self.hparams.temperature
		nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
		nll = nll.mean()
		
		# Logging loss
		self.log(mode+'_loss', nll)
		# Get ranking position of positive example
		comb_sim = torch.cat([cos_sim[pos_mask][:,None], 
							  cos_sim.masked_fill(pos_mask, -9e15)], 
							 dim=-1)
		sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
		#Logging ranking metrics
		self.log(mode+'_acc_top1', (sim_argsort == 0).float().mean(), prog_bar=False, logger=False)
		self.log(mode+'_acc_top5', (sim_argsort < 5).float().mean())
		self.log(mode+'_acc_mean_pos', 1+sim_argsort.float().mean())
		
		return nll
		
	def training_step(self, batch, batch_idx):
		return self.info_nce_loss(batch, mode='train')
		
	def validation_step(self, batch, batch_idx):
		self.info_nce_loss(batch, mode='val')


def main():
	
	parser = get_parser()
	args = parser.parse_args()

	if args.config is not None:
		config_dict = load_yaml_config(args.config)
		args = merge_config_into_args(args, config_dict, parser)


	unlabeled_data = get_simclr_dataset(args.root, args.img_size, args.mean, args.std, args.clip_limit, args.tile_grid_size, labeled = False)
	train_data_contrast = get_simclr_dataset(args.root, args.img_size, args.mean, args.std, args.clip_limit, args.tile_grid_size, labeled = True)
	
	train_loader = get_dataloader(unlabeled_data, args.train_batch, shuffle = True, seed = SEED)
	val_loader = get_dataloader(train_data_contrast, args.train_batch, shuffle = False, seed = SEED)

	model = SimCLR(args.model_name, True,
					   hidden_dim=128, start_lr=args.start_lr, 
					   end_lr=args.end_lr, temperature=0.07, 
					   weight_decay=1e-4, max_epochs=args.num_epochs)		
		
	checkpoint_cb = ModelCheckpoint(
			filename="best_simclr",                    # always best.ckpt
			save_weights_only=True,
			save_top_k=1,
			mode="min",
			monitor="val_loss",
			enable_version_counter=False,       # overwrite instead of best-v1.ckpt
		)
		
	logger = CSVLogger(args.output_dir, name=args.run_name, version=args.version)
		
	device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

	trainer = L.Trainer(logger=logger,
		accelerator="gpu" if str(device).startswith("cuda") else "cpu",
		devices=1,
		max_epochs=args.num_epochs,
		callbacks=[
			checkpoint_cb, 
			LearningRateMonitor('epoch')
		]
	)
	trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
	L.seed_everything(SEED)
	torch.set_float32_matmul_precision('medium')
	os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
	load_dotenv()
	main()