from dotenv import load_dotenv
import os

import lightning as L
import torch
from timm import create_model
from arg_utils import get_parser, load_yaml_config

from dataset_utils import get_dataloader, get_labeled_dataset

import matplotlib.pyplot as plt

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from train_main import load_simclr_v2_backbone


SEED=42
NUM_WORKERS = 4
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def tsne_plots_raw(feats, labels, save_dir):
	emb    = torch.cat(feats).numpy()
	labels = torch.cat(labels).numpy()

	rng = np.random.default_rng(SEED)
	idx = rng.choice(len(emb), size=min(3000, len(emb)), replace=False)

	emb50 = PCA(n_components=50, random_state=0).fit_transform(emb[idx])
	xy = TSNE(n_components=2, perplexity=30, init="pca", random_state=SEED).fit_transform(emb50)

	fig, ax = plt.subplots(1, 1, figsize=(7, 6))
	sc = ax.scatter(xy[:, 0], xy[:, 1], c=labels[idx], s=6, cmap="tab10", vmin=0, vmax=9)
	ax.set_title("Raw Embeddings (True Labels)")
	ax.set_xticks([]); ax.set_yticks([])
	plt.colorbar(sc, ax=ax, ticks=range(6))
	
	plt.savefig(os.path.join(save_dir, "tsne_plot_raw_temp01.png"))
	plt.close()

def main():
	parser = get_parser()
	
	args, _ = parser.parse_known_args()

	if args.config is not None:
		config_dict = load_yaml_config(args.config)
		parser.set_defaults(**config_dict)
	args = parser.parse_args()

	os.makedirs(args.output_dir, exist_ok=True)

	val_data = get_labeled_dataset(args.root, "valid", args.img_size, args.mean, args.std, args.clip_limit, args.tile_grid_size)
	
	backbone = load_simclr_v2_backbone(args.checkpoint_path, args.model_name, keep_head_layer = False)

	base_network = backbone
	# base_network = create_model(
	# 							args.model_name, 
	# 							pretrained=args.imagenet_weights,
	# 							num_classes=128)
	base_network.eval().to(device)

	raw_feats, raw_labels = [], []
	
	val_loader = get_dataloader(val_data, args.val_batch, shuffle = False, seed = SEED)

	with torch.no_grad():
		for x, y in val_loader:
			h = base_network(x.to(device))
			raw_feats.append(h.cpu())
			raw_labels.append(y)
			
	log_dir_path = args.results_folder + "/embeddings"
	os.makedirs(log_dir_path, exist_ok=True)
	tsne_plots_raw(raw_feats, raw_labels, save_dir=log_dir_path)

if __name__ == "__main__":
	L.seed_everything(SEED, workers=True)
	os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
	torch.set_float32_matmul_precision('medium')
	load_dotenv()
	main()
