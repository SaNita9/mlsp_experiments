import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision

def show_images(unlabeled_data):
	NUM_IMAGES = 6
	imgs = torch.stack([img for idx in range(NUM_IMAGES) for img in unlabeled_data[idx][0]], dim=0)
	img_grid = torchvision.utils.make_grid(imgs, nrow=6, normalize=True, pad_value=0.9)
	img_grid = img_grid.permute(1, 2, 0)

	plt.figure(figsize=(10,5))
	plt.title('Augmented image examples of the OIA_DDR dataset')
	plt.imshow(img_grid)
	plt.axis('off')
	plt.show()
	plt.close()

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