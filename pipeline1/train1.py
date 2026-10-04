import torch
import os
from timm import create_model
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim.lr_scheduler as lr_scheduler
from dotenv import load_dotenv
from typing import Tuple, Callable
from torch.utils.data import DataLoader
import math


from dataset1 import get_dataloaders, get_resampled_dataloaders, get_datasets
from utils1 import make_deterministic
from utils1 import get_parser, load_yaml_config, merge_config_into_args 
from utils1 import see_distribution

device = 'cuda' if torch.cuda.is_available() else 'cpu'

SEED = 42

class DDRModel(nn.Module):
	def __init__(self, model_name: str = "resnet18", pretrained: bool = True, num_classes: int = 6):
			super().__init__()

			self.model = create_model(
				model_name, 
				pretrained=True,
				num_classes=num_classes
			)

	def forward(self, x):
		return self.model(x)

def train_epoch(model: nn.Module, 
				loader: DataLoader, 
				optimizer: torch.optim.Optimizer, 
				criterion: nn.Module, 
				scaler: torch.amp.GradScaler) -> Tuple[float, float]:
	
	model.train()
	total_loss, correct = 0, 0
	for images, labels in loader:
		images = images.to(device, non_blocking=True)
		labels = labels.to(device, non_blocking=True)
		optimizer.zero_grad()
		with torch.amp.autocast('cuda'):
			outputs = model(images)
			loss = criterion(outputs, labels)

		scaler.scale(loss).backward()
		scaler.step(optimizer)
		scaler.update()
		total_loss += loss.item()
		correct += (outputs.argmax(1) == labels).sum().item()
	return total_loss / len(loader), correct / len(loader.dataset)

def validate(model: nn.Module, 
             loader: DataLoader, 
			 criterion: nn.Module) -> Tuple[float, float]:
	
	model.eval()
	total_loss, correct = 0, 0
	with torch.no_grad():
		for images, labels in loader:
			images = images.to(device, non_blocking=True)
			labels = labels.to(device, non_blocking=True)
			outputs = model(images)
			total_loss += criterion(outputs, labels).item()
			correct += (outputs.argmax(1) == labels).sum().item()
	return total_loss / len(loader), correct / len(loader.dataset)

def execution_loop(checkpoint_path: str,
					model: nn.Module,
					trainloader: DataLoader,
					valloader: DataLoader,
					criterion: nn.Module,
					optimizer: torch.optim.Optimizer,
					scaler: torch.amp.GradScaler,
					lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
					epochs: int,
					training_fn: Callable,
					print_stats: bool = True):

	if print_stats:
		print(f"{'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  {'Val Loss':>8}  {'Val Acc':>7}")
		print("-" * 52)
	tr_loss_values = []
	vl_loss_values = []
	tr_batch_acc_values = []
	vl_batch_acc_values = []
	
	for epoch in range(epochs):
		
		tr_loss, tr_acc = training_fn(model, trainloader, optimizer, criterion, scaler)
		tr_loss_values.append(tr_loss)
		tr_batch_acc_values.append(tr_acc)

		vl_loss, vl_acc = validate(model, valloader, criterion)

		if epoch != 0 and vl_loss_values[-1] > vl_loss:
			checkpoint = {
			'epoch': epoch,
			'model_state_dict': model.state_dict(),
			'optimizer_state_dict': optimizer.state_dict(),
			'loss': tr_loss
			}
			torch.save(checkpoint, checkpoint_path)
			print(f"Overwritten checkpoint at epoch {epoch}")

		vl_loss_values.append(vl_loss)
		vl_batch_acc_values.append(vl_acc)

		if print_stats:
			print(f"{epoch+1:>5}  {tr_loss:>10.4f}  {tr_acc:>9.4f}  {vl_loss:>8.4f}  {vl_acc:>7.7f}")
		
		
		lr_scheduler.step()
		

	plt.plot(range(epochs), tr_loss_values, label = "train loss")
	plt.plot(range(epochs), vl_loss_values, label = "val loss")
	plt.legend()
	plt.title("Loss accross epochs")
	plt.show()

	plt.plot(range(epochs), tr_batch_acc_values, label = "train acc")
	plt.plot(range(epochs), vl_batch_acc_values, label = "val acc")
	plt.legend()
	plt.title("Accuracy accross epochs")
	plt.show()

def main():
	parser = get_parser()
	args = parser.parse_args()

	if args.config is not None:
		config_dict = load_yaml_config(args.config)
		args = merge_config_into_args(args, config_dict, parser)

	model = DDRModel(args.model_name, pretrained=True, num_classes=6)
	model.to(device)

	optim = torch.optim.Adam(model.parameters(), lr=args.start_lr)
	scheduler = lr_scheduler.CosineAnnealingLR(optim, T_max=args.num_epochs, eta_min=args.end_lr)

	trainset = get_datasets(args.root, 'train', args.img_size, args.mean, args.std)
	valset = get_datasets(args.root, 'valid', args.img_size, args.mean, args.std)
	
	trainloader = get_resampled_dataloaders(trainset, args.train_batch, shuffle = True, seed = SEED)
	valloader = get_dataloaders(valset, args.val_batch, shuffle = False, seed = SEED)

	# see_distribution(trainloader)

	criterion = nn.CrossEntropyLoss()
	scaler = torch.amp.GradScaler('cuda')

	execution_loop( args.checkpoint_path, model,
					trainloader, valloader,
					criterion, optim, scaler,
					args.train_batch, args.val_batch,
					scheduler, args.num_epochs,
					train_epoch,
					print_stats=args.print_stats)
	
if __name__ == "__main__":
	make_deterministic(SEED)
	os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
	load_dotenv()
	main()
