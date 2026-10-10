import time
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
import torchvision.transforms as transforms
from utils1 import seed_worker
import torchvision.io as io
from torchvision.transforms import v2
import cv2

class DDR_Dataset(Dataset):
	def __init__(self, root, mode, transform=None, small=True):
		if small:
			dir_name = mode + "512"
		else:
			dir_name = mode
		self.img_dir = os.path.join(root, dir_name)
		self.transform = transform
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
		t0 = time.time()
		img_name, label = self.data[index]
		img_path = os.path.join(self.img_dir, img_name)

		img = io.read_image(img_path)

		if self.transform:
			img = self.transform(img)

		return img, label
class ApplyCLAHE:
	def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
		self.clip_limit = clip_limit
		self.tile_grid_size = tile_grid_size

	def __call__(self, img_tensor: torch.Tensor) -> torch.Tensor:
		if img_tensor.dtype != torch.uint8:
			raise TypeError(f"ApplyCLAHE expects torch.uint8, but got {img_tensor.dtype}. "
							"Ensure ToDtype(torch.float32) happens after this transform.")

		#PyTorch uses CHW (Channels, Height, Width), while OpenCV expects HWC (Height, Width, Channels)
		img_np = img_tensor.permute(1, 2, 0).numpy()
		lab_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
	
		L,A,B=cv2.split(lab_img)
		clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
		clahe_L = np.clip(clahe.apply(L))
	
		clahe_lab = cv2.merge((clahe_L, A, B))
		clahe_rgb = cv2.cvtColor(clahe_lab, cv2.COLOR_LAB2RGB)
		out_tensor = torch.from_numpy(clahe_rgb).permute(2, 0, 1)
		return out_tensor

def get_datasets(root, stage, img_size, mean, std, clip_limit, tile_grid_size):

	transform_dict = {
	'train': v2.Compose([
			#CropBackground(threshold=5),
			ApplyCLAHE(clip_limit=clip_limit, tile_grid_size=tile_grid_size),
			v2.Resize((img_size, img_size)),
			#rotation? -15 to 15
			v2.RandomAffine(
				# translation,
				translate=(0.1, 0.1),
				# stretching,
				scale=(0.9, 1.1), 
				# rotation,
				degrees=15,
				
				fill=0
			),
			# flipping, 
			v2.RandomHorizontalFlip(),
			# and colour augmentation
			v2.ColorJitter(brightness=0.5, contrast=1, saturation=0.1, hue=0.25),
	
			# transforms.ToTensor(),
			v2.ToDtype(torch.float32, scale=True),
			v2.Normalize(mean, std)
		]),
	
		'eval' : transforms.Compose([
			#CropBackground(threshold=5),
			ApplyCLAHE(clip_limit=clip_limit, tile_grid_size=tile_grid_size),
			v2.Resize((img_size, img_size)),
			# transforms.ToTensor(),
			v2.ToDtype(torch.float32, scale=True),
			v2.Normalize(mean, std)
		])
	}

	if stage == 'train':
		transform = transform_dict['train']
	else:
		transform = transform_dict['eval']
	dataset = DDR_Dataset(root, stage, transform=transform)
	
	return dataset

def get_dataloaders(dataset, batch_size, shuffle=False , seed=42):
	g = torch.Generator()
	g.manual_seed(seed)
	loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
						num_workers=4, worker_init_fn=seed_worker, 
						generator=g, pin_memory=True)
	return loader

def get_resampled_dataloaders(dataset, batch_size, shuffle=False , seed=42):
	
	g = torch.Generator()
	g.manual_seed(seed)
	
	labels_cpu = np.array([label for _, label in dataset.data])
	class_counts = np.bincount(labels_cpu, minlength=6)
	class_weights = 1/class_counts
	sample_weights = [class_weights[i] for i in labels_cpu]
	
	sampler= WeightedRandomSampler(weights=sample_weights, num_samples=len(labels_cpu), replacement=True)
	loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, 
						num_workers=4, worker_init_fn=seed_worker, 
						generator=g, pin_memory=True)
	return loader