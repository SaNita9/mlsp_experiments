import torch
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import torch.nn as nn
from timm import create_model
import matplotlib.pyplot as plt

from dataset1 import get_dataloaders, get_resampled_dataloaders, get_datasets
from utils1 import set_seed, make_deterministic

device = 'cuda' if torch.cuda.is_available() else 'cpu'

from sklearn.metrics import cohen_kappa_score

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

def run_inference(model_name, num_epochs, start_lr, end_lr, dataloader, print_stats):

	#aici ar trebui sa-l fac determinist
	
	model = create_model(model_name, pretrained=True, num_classes=6)
	#aici presupun ca ia de la dataloader
	model.to(device)
	criterion = nn.CrossEntropyLoss()
	
	test_loss, overall_test_accuracy, test_acc_sep, kappa = evaluate(model, testloader, criterion, confusion_matrix_show=True)
	average_test_accuracy = np.sum(test_acc_sep) / 6
	print(f"\nTest metrics:  Loss {test_loss:>8.4f}  OA {overall_test_accuracy:.4f}  AA {average_test_accuracy:.4f}, kappa {kappa:>.4f}")
	num_grades = 6
	for i in range(num_grades):
		print(f"{i}: {test_acc_sep[i]:>10.4f}")

def main():
    run_inference('resnet18', 50, 1e-4, 1e-7, get_resampled_dataloaders, print_stats = True)
    
if __name__ == "__main__":
    main()
