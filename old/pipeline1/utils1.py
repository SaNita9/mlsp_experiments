import random
import torch
import os
import numpy as np
import matplotlib.pyplot as plt

import argparse
from pathlib import Path
from typing import Any, Dict

def get_parser():
	parser = argparse.ArgumentParser(description="trial")
	parser.add_argument("--config", type=str, default=None)
	parser.add_argument("--patience", type=int, default=6)
	parser.add_argument("--delta", type=float, default=0.01)
	parser.add_argument("--img_size", type=int, default=224)
	parser.add_argument("--mean", type=float, nargs='+', default=[0.4140, 0.2575, 0.1289])
	parser.add_argument("--std", type=float, nargs='+', default=[0.2945, 0.2047, 0.1401])
	parser.add_argument("--num_epochs", type=int, default=5)
	parser.add_argument("--train_batch", type=int, default=64)
	parser.add_argument("--test_batch", type=int, default=128)
	parser.add_argument("--val_batch", type=int, default=128)
	parser.add_argument("--start_lr", type=float, default=1e-4)
	parser.add_argument("--end_lr", type=float, default=1e-7)
	parser.add_argument("--print_stats", action=argparse.BooleanOptionalAction, default=True)
	parser.add_argument("--model_name", type=str, default="resnet18")
	parser.add_argument("--root", type=str, default=None)
	parser.add_argument("--checkpoint_path", type=str, default=None)	
	parser.add_argument("--clip_limit", type=float, default=2.0)
	parser.add_argument("--tile_grid_size", type=int,  nargs='+', default=[8,8])
	return parser

def load_yaml_config(path: str | Path) -> Dict[str, Any]:
	try:
		import yaml
	except ImportError as exc:
		raise ImportError(
			"PyYAML is required for --config support. Install it with: pip install pyyaml"
		) from exc

	path = Path(path)
	with path.open("r", encoding="utf-8") as f:
		data = yaml.safe_load(f) or {}

	if not isinstance(data, dict):
		raise ValueError(f"Config file must contain a top-level mapping: {path}")
	return data

def merge_config_into_args(args, config: Dict[str, Any], parser):
	defaults = {
		action.dest: action.default
		for action in parser._actions
		if getattr(action, "dest", None) not in (None, "help")
	}
	for key, value in config.items():
		if not hasattr(args, key):
			continue
		if getattr(args, key) == defaults.get(key):
			setattr(args, key, value)
	return args


def set_seed(seed=42):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

def make_deterministic(seed=42):
	set_seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	torch.use_deterministic_algorithms(True)


def seed_worker(worker_id):
	worker_seed = torch.initial_seed() % 2**32
	np.random.seed(worker_seed)
	random.seed(worker_seed)


def see_distribution(trainloader):

	class_proportions = []
	for _, labels in trainloader:
		labels_cpu = labels.cpu().numpy()
		class_nums = np.bincount(labels_cpu, minlength=6)
		class_proportions.append(class_nums)

	data = np.array(class_proportions)
	num_batches, num_classes = data.shape
	batch_idx = np.arange(num_batches)

	fig, ax = plt.subplots(figsize=(10, 5))
	bottom = np.zeros(num_batches)
	for c in range(num_classes):
		ax.bar(batch_idx, data[:, c], bottom=bottom, label=f'Class {c}')
		bottom += data[:, c]

	ax.set_xlabel('Batch')
	ax.set_ylabel('Sample count')
	ax.set_title('Class distribution per batch')
	ax.legend(title='Class', bbox_to_anchor=(1.05, 1), loc='upper left')
	plt.tight_layout()
	plt.show()