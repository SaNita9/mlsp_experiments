import time
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
import torchvision.transforms as transforms
from utils1 import set_seed, seed_worker ,make_deterministic

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
		img = Image.open(img_path).convert("RGB")
		t1 = time.time()

		if self.transform:
			img = self.transform(img)
		t2 = time.time()
		
		return img, label


def get_datasets(root, stage, img_size, mean, std):

	transform_dict = {
	'train': transforms.Compose([
			#CropBackground(threshold=5),
			transforms.Resize((img_size, img_size)),
			transforms.RandomAffine(
				# translation,
				translate=(0.1, 0.1),
				# stretching,
				scale=(0.9, 1.1), 
				# rotation,
				degrees=15,
				
				fill=0
			),
			# flipping, 
			transforms.RandomHorizontalFlip(),
			# and colour augmentation
			transforms.ColorJitter(brightness=0.5, contrast=1, saturation=0.1, hue=0.25),
	
			transforms.ToTensor(),
			transforms.Normalize(mean, std)
		]),
	
		'eval' : transforms.Compose([
			#CropBackground(threshold=5),
			transforms.Resize((img_size, img_size)),
			transforms.ToTensor(),
			transforms.Normalize(mean, std)
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
	get_dataloaders(dataset, batch_size, shuffle, seed=seed)
	# g = torch.Generator()
	# g.manual_seed(seed)
	
	# labels_cpu = np.array([label for _, label in trainset.data])
	# class_counts = np.bincount(labels_cpu, minlength=6)
	# class_weights = 1/class_counts
	# sample_weights = [class_weights[i] for i in labels_cpu]
	
	# sampler= WeightedRandomSampler(weights=sample_weights, num_samples=len(labels_cpu), replacement=True)

	# trainloader = DataLoader(trainset, batch_size=64, sampler=sampler, num_workers=4, worker_init_fn=seed_worker, generator=g, pin_memory=True)
	# valloader = DataLoader(valset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
	# testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
	# return trainloader,valloader, testloader