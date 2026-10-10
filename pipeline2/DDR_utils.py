import torch
import os
from timm import create_model
from torch.utils.data import Dataset
import lightning as L
import torchvision.io as io

class DDRModel(L.LightningModule):
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