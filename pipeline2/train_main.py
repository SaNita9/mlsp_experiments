from dotenv import load_dotenv
import os
import sys
import lightning as L
import torch
from torchvision.transforms import v2
from timm import create_model
from arg_utils import get_parser, load_yaml_config, merge_config_into_args
import tqdm
from dataset_utils import get_dataloader, get_labeled_dataset, get_resampled_dataloader, get_simclr_dataset
from simclr_pretraining import SimCLR
from model_utils import Classifier
from torch.utils.data import TensorDataset
from lightning.pytorch.loggers import CSVLogger
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

SEED=42
NUM_WORKERS = 4
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def tsne_plots(feats, labels, preds, save_dir):
	emb    = torch.cat(feats).numpy()
	labels = torch.cat(labels).numpy()
	preds  = torch.cat(preds).numpy()

	rng = np.random.default_rng(SEED)
	idx = rng.choice(len(emb), size=min(3000, len(emb)), replace=False)

	emb50 = PCA(n_components=50, random_state=0).fit_transform(emb[idx])
	xy = TSNE(n_components=2, perplexity=30, init="pca", random_state=SEED).fit_transform(emb50)

	fig, ax = plt.subplots(1, 2, figsize=(14, 6))
	for a, c, title in [(ax[0], labels[idx], "true label"), (ax[1], preds[idx], "predicted")]:
		sc = a.scatter(xy[:, 0], xy[:, 1], c=c, s=6, cmap="tab10", vmin=0, vmax=9)
		a.set_title(title); a.set_xticks([]); a.set_yticks([])
	plt.colorbar(sc, ax=ax, ticks=range(6))
	
	# Save instead of show
	plt.savefig(os.path.join(save_dir, "tsne_plot.png"))
	plt.close()


def plot_loss_acc(metrics_dir, log_dir):
	metrics_path = os.path.join(metrics_dir, "metrics.csv")
	
	if not os.path.exists(metrics_path):
		print(f"No metrics found at {metrics_path}")
		return
		
	metrics = pd.read_csv(metrics_path)
	print("Metrics columns:", metrics.columns.tolist())
		
	metrics_by_epoch = metrics.groupby("epoch").mean().reset_index()

	fig, axes = plt.subplots(1, 2, figsize=(14, 5))

	axes[0].plot(metrics_by_epoch["epoch"], metrics_by_epoch["train_loss_epoch"], label="Train Loss", marker="o")
	axes[0].plot(metrics_by_epoch["epoch"], metrics_by_epoch["val_loss"], label="Val Loss", marker="o")
	axes[0].set_title("Training and Validation Loss")
	axes[0].set_xlabel("Epoch")
	axes[0].set_ylabel("Loss")
	axes[0].legend()
	axes[0].grid(True)

	axes[1].plot(metrics_by_epoch["epoch"], metrics_by_epoch["train_acc_epoch"], label="Train Accuracy", marker="o")
	axes[1].plot(metrics_by_epoch["epoch"], metrics_by_epoch["val_acc"], label="Val Accuracy", marker="o")
	axes[1].set_title("Training and Validation Accuracy")
	axes[1].set_xlabel("Epoch")
	axes[1].set_ylabel("Accuracy")
	axes[1].legend()
	axes[1].grid(True)

	plt.tight_layout()
	
	plt.savefig(os.path.join(log_dir, "loss_acc_plot.png"))
	plt.close()


def prepare_features_logistic_reg(network, dataset, batch_size, seed):
	network.eval()
	network.to(device)
	data_loader = get_dataloader(dataset, batch_size, shuffle=False, seed=seed)
	feats, labels = [], []
	with torch.no_grad():     
		for batch_imgs, batch_labels in tqdm.tqdm(data_loader):
			batch_imgs = batch_imgs.to(device)
			batch_feats = network(batch_imgs)
			feats.append(batch_feats.detach().cpu())
			labels.append(batch_labels)
	
	feats = torch.cat(feats)
	labels = torch.cat(labels)

	return TensorDataset(feats, labels)

def load_simclr_v2_backbone(checkpoint_path, model_name, keep_head_layer=True):
	"""
	Rebuild the timm backbone and optionally attach the first layer of the projection head,
	which is a standard SimCLR v2 practice for fine-tuning.
	"""
	ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
	state_dict = ckpt.get("state_dict", ckpt)

	suffix = "conv1.weight"
	candidates = [k for k in state_dict if k.endswith(suffix) and "projection" not in k]
	if not candidates:
		raise KeyError(f"No '{suffix}' key found.")

	prefix = min(candidates, key=len)[:-len(suffix)]
	backbone_state = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix) and "projection" not in k}

	backbone = create_model(model_name, pretrained=False, num_classes=0)
	backbone.load_state_dict(backbone_state, strict=True)
	
	if not keep_head_layer:
		return backbone
		
	head_state = {k: v for k, v in state_dict.items() if "projection_head.0" in k or "projection_head.1" in k}
	
	first_layer = nn.Sequential(
		nn.Linear(backbone.num_features, backbone.num_features, bias=False), 
		nn.BatchNorm1d(backbone.num_features),
		nn.ReLU(inplace=True)
	)

	cleaned_head_state = {}
	for k, v in head_state.items():
		if "projection_head.0" in k:
			cleaned_head_state[k.split("projection_head.0.")[-1]] = v
		elif "projection_head.1" in k:
			cleaned_head_state["1." + k.split("projection_head.1.")[-1]] = v
			
	first_layer[0].load_state_dict({k: v for k, v in cleaned_head_state.items() if not k.startswith("1.")})
	first_layer[1].load_state_dict({k.replace("1.", ""): v for k, v in cleaned_head_state.items() if k.startswith("1.")})

	combined_model = nn.Sequential(
		backbone,
		first_layer
	)
	return combined_model

def main():
	parser = get_parser()
	
	args, _ = parser.parse_known_args()

	if args.config is not None:
		config_dict = load_yaml_config(args.config)
		parser.set_defaults(**config_dict)
	args = parser.parse_args()

	os.makedirs(args.output_dir, exist_ok=True)
	
	cb_file_name = "best_" + str(args.pretraining_type) 
	if args.pretraining_type == 'SimCLR':
		cb_file_name += str(args.embeddings_only)
	checkpoint_cb = ModelCheckpoint(
		filename=cb_file_name,
		dirpath=args.output_dir,
		save_weights_only=True,
		save_top_k=1,
		mode="min",
		monitor="val_loss",
		enable_version_counter=False,
	)

	early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=10, verbose=False, mode="min")
	
	logger = CSVLogger(args.output_dir, name="", version="")
	
	trainer = L.Trainer(logger=logger,
						accelerator="gpu" if str(device).startswith("cuda") else "cpu",
						devices=1,
						max_epochs=args.num_epochs,
						callbacks=[checkpoint_cb, early_stop_callback,
								LearningRateMonitor("epoch")],
						enable_progress_bar=True,
						check_val_every_n_epoch=1,
						deterministic=True,)

	train_data = get_labeled_dataset(args.root, "train", args.img_size, args.mean, args.std, args.clip_limit, args.tile_grid_size)
	val_data = get_labeled_dataset(args.root, "valid", args.img_size, args.mean, args.std, args.clip_limit, args.tile_grid_size)
	test_data = get_labeled_dataset(args.root, "test", args.img_size, args.mean, args.std, args.clip_limit, args.tile_grid_size)
	
	match args.pretraining_type:
		case "none": 
			base_network = create_model(
							args.model_name, 
							pretrained=args.imagenet_weights,
							num_classes=0
						)
			resampled = args.resample
			feature_dim = None
			freeze_backbone = False

		case "SimCLR":
			backbone = load_simclr_v2_backbone(args.checkpoint_path, args.model_name, keep_head_layer = False)
			
			if args.embeddings_only:
				train_data = prepare_features_logistic_reg(backbone, dataset = train_data, batch_size = args.train_batch, seed = SEED)
				val_data = prepare_features_logistic_reg(backbone, dataset =val_data, batch_size = args.val_batch, seed = SEED)
				test_data = prepare_features_logistic_reg(backbone, dataset =test_data, batch_size = args.test_batch, seed = SEED)
				backbone.cpu()
				base_network = nn.Identity()
				feature_dim = train_data.tensors[0].shape[1]
				resampled = False
				freeze_backbone = True

			else: 
				base_network = backbone
				resampled = args.resample
				feature_dim = None
				freeze_backbone = False

		case _:
			raise ValueError(f"Unknown pretraining_type: {args.pretraining_type}")

	print(len(train_data), len(val_data), len(test_data)) 

	if resampled:
		train_loader = get_resampled_dataloader(train_data, args.train_batch, shuffle=False, seed = SEED)
	else:
		train_loader = get_dataloader(train_data, args.train_batch, shuffle = True, seed = SEED)
	
	val_loader = get_dataloader(val_data, args.val_batch, shuffle = False, seed = SEED)
	test_loader = get_dataloader(test_data, args.test_batch, shuffle = False, seed = SEED)

	log_dir_path = args.results_folder + "/" + str(args.pretraining_type)
	if args.pretraining_type == 'SimCLR':
			log_dir_path += str(args.embeddings_only)
	log_dir_path += str(args.num_epochs)

	backbone_lr_mult = 1.0 if args.pretraining_type == "none" else 0.1

	model = Classifier(
					base_network, 
					feature_dim= feature_dim,
					num_classes=6,
					weight_decay=1e-4,
					betas = (0.9, 0.999),
					start_lr = args.start_lr,
					end_lr = args.end_lr,
					eps = args.eps,
					warmup_epochs=0, 
					backbone_lr_mult=backbone_lr_mult, 
					freeze_backbone=freeze_backbone,
					max_epochs=args.num_epochs, label_smoothing=0.0,
					output_dir=log_dir_path
					)

	trainer.fit(model, train_loader, val_loader)

	best_ckpt_path = trainer.checkpoint_callback.best_model_path
	if best_ckpt_path:
		print(f"Loading best model {best_ckpt_path}")
		checkpoint = torch.load(best_ckpt_path, map_location=device)
		model.load_state_dict(checkpoint['state_dict'])

	model.eval().to(device)

	feats, labels, preds = [], [], []
	with torch.no_grad():
		for x, y in val_loader:      
			h = model.backbone(x.to(device)) 
			feats.append(h.cpu())
			preds.append(model.head(h).argmax(1).cpu())
			labels.append(y)

	# Save figures to the lightning log directory
	
	
	tsne_plots(feats, labels, preds, save_dir=log_dir_path)
	plot_loss_acc(args.output_dir, log_dir_path)

	trainer.test(model, test_loader, ckpt_path=best_ckpt_path)  
	print("Best checkpoint path:", best_ckpt_path)

if __name__ == "__main__":
	L.seed_everything(SEED, workers=True)
	os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
	torch.set_float32_matmul_precision('medium')
	load_dotenv()
	main()
