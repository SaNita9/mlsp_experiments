import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torchvision
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import torchvision.io as io
from torchvision.transforms import v2
from timm import create_model
from dotenv import load_dotenv

from dataset1 import get_dataloaders, get_resampled_dataloaders, get_datasets
from utils1 import make_deterministic
from utils1 import get_parser, load_yaml_config, merge_config_into_args 
from train1 import DDRModel

from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

SEED = 42

"""
source: https://github.com/phlippe/uvadlc_notebooks/blob/master/docs/tutorial_notebooks/tutorial17/SimCLR.ipynb
"""

class DDR_Dataset(Dataset):
	def __init__(self, root, mode, transform=None, small=True, labeled = True):
		if small:
			dir_name = mode + "512"
		else:
			dir_name = mode
		self.img_dir = os.path.join(root, dir_name)
		self.transform = transform
		self.labeled = labeled
		if self.labeled == False:
			print(dir_name)
			self.data = os.listdir(self.img_dir)
		else:
			self.data = []
			labels_file = mode + ".txt"
			path = os.path.join(root, labels_file)
			with open(path, 'r') as f:
				for line in f:
					line = line.strip()
					if not line:
						continue
					img_name, grade = line.split()
					self.data.append((img_name, int(grade)))


	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		if self.labeled:
			img_name, label = self.data[index]
		else:
			img_name = self.data[index]
			label = -1

		img_path = os.path.join(self.img_dir, img_name)
		img = io.read_image(img_path)

		if self.transform:
			img = self.transform(img)
			
		return img, label


class ContrastiveTransformations(object):
	
	def __init__(self, base_transforms, n_views=2):
		self.base_transforms = base_transforms
		self.n_views = n_views
		
	def __call__(self, x):
		return [self.base_transforms(x) for i in range(self.n_views)]


class SimCLR(pl.LightningModule):
	
	def __init__(self, model_name, pretrained, num_classes, hidden_dim, lr, temperature, weight_decay, max_epochs=50):
		super().__init__()
		self.save_hyperparameters()
		
		self.convnet = DDRModel(model_name, pretrained=pretrained, num_classes=0)
	   
		backbone_out_features = self.convnet.num_features
		
		# 3. Create the SimCLR projection head separately
		self.projection_head = nn.Sequential(
			nn.Linear(backbone_out_features, backbone_out_features), 
			nn.ReLU(inplace=True),
			nn.Linear(backbone_out_features, hidden_dim)
		)

	def configure_optimizers(self):
		optimizer = optim.AdamW(self.parameters(), 
								lr=self.hparams.lr, 
								weight_decay=self.hparams.weight_decay)
		lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
															T_max=self.hparams.max_epochs,
															eta_min=self.hparams.lr)
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
		self.log(mode+'_loss', nll)
		
		return nll
		
	def training_step(self, batch, batch_idx):
		return self.info_nce_loss(batch, mode='train')
		
	def validation_step(self, batch, batch_idx):
		self.info_nce_loss(batch, mode='val')


def show_images(unlabeled_data):
	NUM_IMAGES = 6
	imgs = torch.stack([img for idx in range(NUM_IMAGES) for img in unlabeled_data[idx][0]], dim=0)
	img_grid = torchvision.utils.make_grid(imgs, nrow=6, normalize=True, pad_value=0.9)
	img_grid = img_grid.permute(1, 2, 0)

	plt.figure(figsize=(10,5))
	plt.title('Augmented image examples of the OIA_DDR dataset')
	plt.imshow(img_grid)
	plt.axis('off')
	plt.show()
	plt.close()

def main():

	NUM_WORKERS = 4

	parser = get_parser()
	args = parser.parse_args()

	if args.config is not None:
		config_dict = load_yaml_config(args.config)
		args = merge_config_into_args(args, config_dict, parser)

	pl.seed_everything(42)
	device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
	print("Device:", device)
	print("Number of workers:", NUM_WORKERS)
	img_size = args.img_size
	contrast_transforms = v2.Compose([v2.Resize((img_size, img_size)),
										v2.RandomHorizontalFlip(),
										  v2.RandomResizedCrop(size=args.img_size),
										  v2.RandomApply([
											  v2.ColorJitter(brightness=0.5, 
																	 contrast=0.5, 
																	 saturation=0.5, 
																	 hue=0.1)
										  ], p=0.8),
										  v2.RandomGrayscale(p=0.2),
										  v2.GaussianBlur(kernel_size=9),
										  v2.ToDtype(torch.float32, scale=True),
										  v2.Normalize(args.mean, args.std)
										 ])

	unlabeled_data = DDR_Dataset(args.root, 'train', ContrastiveTransformations(contrast_transforms, n_views=2), labeled=False)
	train_data_contrast = DDR_Dataset(args.root, 'train', ContrastiveTransformations(contrast_transforms, n_views=2), labeled=True)
	train_loader = data.DataLoader(unlabeled_data, batch_size=args.train_batch, shuffle=True, 
									drop_last=True, pin_memory=True, num_workers=NUM_WORKERS)
	val_loader = data.DataLoader(train_data_contrast, batch_size=args.val_batch, shuffle=False, 
	                             drop_last=False, pin_memory=True, num_workers=NUM_WORKERS)

	model = SimCLR(args.model_name, True, 6, hidden_dim=128, lr=5e-4, temperature=0.07, weight_decay=1e-4, max_epochs=args.num_epochs)		
	
	save_dir = os.path.dirname(args.checkpoint_path)
	file_name = os.path.splitext(os.path.basename(args.checkpoint_path))[0]

	checkpoint_callback = ModelCheckpoint(
		dirpath=save_dir,
		filename=file_name,
		save_top_k=1,
		mode='min',  
        monitor='val_loss',
		save_weights_only=True
	)

	# Setup Trainer
	trainer = pl.Trainer(
		default_root_dir=save_dir,
		accelerator="gpu" if str(device).startswith("cuda") else "cpu",
		devices=1,
		max_epochs=args.num_epochs,
		callbacks=[
			checkpoint_callback, 
			LearningRateMonitor('epoch')
		]
	)
	trainer.fit(model, train_loader, val_loader)
		

if __name__ == "__main__":
	pl.seed_everything(SEED)
	torch.set_float32_matmul_precision('medium')
	os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
	load_dotenv()
	main()