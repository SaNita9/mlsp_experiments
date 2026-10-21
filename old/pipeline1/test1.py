import torch
import os
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import torch.nn as nn
import matplotlib.pyplot as plt
from train1 import DDRModel
from dotenv import load_dotenv
from utils1 import get_parser, load_yaml_config, merge_config_into_args 


from dataset1 import get_dataloaders, get_resampled_dataloaders, get_datasets
from utils1 import set_seed, make_deterministic

device = 'cuda' if torch.cuda.is_available() else 'cpu'

from sklearn.metrics import cohen_kappa_score

SEED = 42

def evaluate(model, loader, criterion, confusion_matrix_show=False):
	model.eval()
	total_loss, correct = 0, 0
	correct_per_class = np.zeros(6)
	class_nums = np.zeros(6)
	all_labels = []
	all_preds = []
	with torch.no_grad():
		for images, labels in loader:
			images = images.to(device, non_blocking=True)
			labels = labels.to(device, non_blocking=True)
			outputs = model(images)
			total_loss += criterion(outputs, labels).item()
			preds = outputs.argmax(1)

			labels_np = labels.cpu().numpy()
			preds_np = preds.cpu().numpy()

			class_nums += np.bincount(labels_np, minlength=6)
			
			correct_mask = (preds_np == labels_np)
			correct_per_class += np.bincount(labels_np[correct_mask], minlength=6)
			correct += (outputs.argmax(1) == labels).sum().item()

			all_labels.append(labels_np)
			all_preds.append(preds_np)

		all_labels = np.concatenate(all_labels)
		all_preds = np.concatenate(all_preds) 
		
		per_class_acc = np.divide(
			correct_per_class, class_nums,
			out=np.zeros_like(correct_per_class), where=class_nums != 0
			)

		kappa = cohen_kappa_score(all_preds, all_labels)
		
		if confusion_matrix_show:
			cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2, 3, 4, 5])
			disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1, 2, 3, 4, 5])
			disp.plot(cmap='Blues', values_format='d')
			plt.title("Confusion Matrix")
			plt.show()
		return total_loss / len(loader), correct / len(loader.dataset), per_class_acc, kappa

def run_test(test_batch, root, img_size, mean, std, clip_limit, tile_grid_size, model_name, dataloader_fn):

	make_deterministic(SEED)
	
	os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
	checkpoint_path = "/home/sara/distilaimedical/pipeline1/checkpoint.pth"
	
	model = DDRModel(model_name, pretrained=False, num_classes=6)
	model.to(device)
	
	
	criterion = nn.CrossEntropyLoss()

	checkpoint = torch.load(checkpoint_path, weights_only=True)
	model.load_state_dict(checkpoint['model_state_dict'])

	testset = get_datasets(root, 'test', img_size, mean, std, clip_limit, tile_grid_size)
	testloader = dataloader_fn(testset, test_batch, shuffle = True, seed = SEED)

	test_loss, overall_test_accuracy, test_acc_sep, kappa = evaluate(model, testloader, criterion, confusion_matrix_show=True)
	average_test_accuracy = np.sum(test_acc_sep) / 6
	print(f"\nTest metrics:  Loss {test_loss:>8.4f}  OA {overall_test_accuracy:.4f}  AA {average_test_accuracy:.4f}, kappa {kappa:>.4f}")
	num_grades = 6
	for i in range(num_grades):
		print(f"{i}: {test_acc_sep[i]:>10.4f}")

def main():
	parser = get_parser()
	args = parser.parse_args()

	if args.config is not None:
		config_dict = load_yaml_config(args.config)
		args = merge_config_into_args(args, config_dict, parser)

	run_test(args.test_batch, args.root, args.img_size, args.mean, args.std, args.clip_limit, args.tile_grid_size, args.model_name, get_resampled_dataloaders)
	
if __name__ == "__main__":
	load_dotenv()
	main()
