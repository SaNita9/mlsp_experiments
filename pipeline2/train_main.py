from dotenv import load_dotenv
import os
import lightning as L
import torch
from torchvision.transforms import v2
from timm import create_model
from arg_utils import get_parser, load_yaml_config, merge_config_into_args
from pipeline2.model_utils import DDRModel
from dataset_utils import ApplyCLAHE, DDR_Dataset, get_dataloader, get_labeled_dataset, get_resampled_dataloader, get_simclr_dataset
from simclr_pretraining import SimCLR
from model_utils import ConvModel, LogisticRegression, Model, Pretrained_Model
from torch.utils.data import Dataset, DataLoader, TensorDataset
from copy import deepcopy
import tqdm
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
SEED=42
NUM_WORKERS = 4
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def prepare_features_logistic_reg(network, dataset, batch_size, seed):
	network.eval()
	network.to(device)
	data_loader = get_dataloader(dataset, batch_size, seed)
	feats, labels = [], []
	for batch_imgs, batch_labels in tqdm.tqdm(data_loader):
		batch_imgs = batch_imgs.to(device)
		batch_feats = network(batch_imgs)
		feats.append(batch_feats.detach().cpu())
		labels.append(batch_labels)
	
	feats = torch.cat(feats, dim=0)
	labels = torch.cat(labels, dim=0)
	
	# Sort images by labels
	labels, idxs = labels.sort()
	feats = feats[idxs]
	return TensorDataset(feats, labels)


def main():
	parser = get_parser()
	args = parser.parse_args()

	if args.config is not None:
		config_dict = load_yaml_config(args.config)
		args = merge_config_into_args(args, config_dict, parser)

	"""
		nimic + conv + nimic
		nimic + self sup contrastive learning 1) 2) 3) + nimic
		nimic + sup contrastive learning + nimic
	"""

	"""
		imagenet + conv + nimic
		imagenet + self sup contrastive learning + nimic
		imagenet + sup contrastive learning + nimic
	"""

	"""
		domian-specific/other + conv + nimic
		domian-specific/other + self sup contrastive learning + nimic
		domian-specific/other + sup contrastive learning + nimic
	"""

	"""
		self sup contrastive learning 1) 2) 3) + conv
		sup contrastive learning + conv
	"""

	save_dir = os.path.dirname(args.checkpoint_path)
	file_name = os.path.splitext(os.path.basename(args.checkpoint_path))[0]
	trainer = L.Trainer(default_root_dir=os.path.join(save_dir, file_name),
						accelerator="gpu" if str(device).startswith("cuda") else "cpu",
						devices=1,
						max_epochs=args.num_epochs,
						callbacks=[ModelCheckpoint(save_weights_only=True, mode='min', monitor='val_loss'),
								LearningRateMonitor("epoch")],
						enable_progress_bar=False,
						check_val_every_n_epoch=1)
	
	trainer.logger._default_hp_metric = None

	train_data = get_labeled_dataset(args.root, "train", args.img_size, args.mean, args.std, args.clip_limit, args.tile_grid_size)
	val_data = get_labeled_dataset(args.root, "valid", args.img_size, args.mean, args.std, args.clip_limit, args.tile_grid_size)
	
	match args.pretraining_type:
		case "none": #VANILLA RESNET
			# model = ConvModel(args.model_name, pretrained=args.imagenet_weights, num_classes=6)
			architecture = create_model(
							args.model_name, 
							pretrained=args.imagenet_weights,
							num_classes=6
						)
			resampled = args.resample

		case "SimCLR":
			# simclr_model = backbone and projection head
			simclr_model = SimCLR.load_from_checkpoint(args.checkpoint_path)
			simclr_model.eval()
			with torch.no_grad():
				backbone = deepcopy(simclr_model.convnet)

			
			if args.model_name == "JUST_SIMCLR_EMBEDDINGS":
				train_data = prepare_features_logistic_reg(backbone, dataset = train_data, batch_size = args.train_batch, shuffle=False, seed = SEED)
				val_data = prepare_features_logistic_reg(simclr_model, dataset =val_data, batch_size = args.val_batch, shuffle=False, seed = SEED)
				model = LogisticRegression(feature_dim=train_data.tensors[0].shape[1],
											num_classes=6,
											lr=1e-3,
											weight_decay=1e-3)

			else: 
				model = backbone
				resampled = args.resample
			
			resampled = False


	if resampled == True:
		train_loader = get_resampled_dataloader(train_data, args.train_batch, shuffle=False, seed = SEED)
	else:
		train_loader = get_dataloader(train_data, args.train_batch, seed = SEED)
	
	val_loader = get_dataloader(val_data, args.train_batch, seed = SEED)
	model = Model(architecture, feature_dim=train_data.tensors[0].shape[1],
												num_classes=6,
												lr=1e-3,
												weight_decay=1e-3)
	model.to(device)
	model.fit(model, train_loader, val_loader)
	##NEED TO ADD CHECKPOINT LOGIC

if __name__ == "__main__":
	
	L.seed_everything(SEED)
	os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
	load_dotenv()
	main()