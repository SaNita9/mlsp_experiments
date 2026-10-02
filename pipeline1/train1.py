import torch
from timm import create_model
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim.lr_scheduler as lr_scheduler

from dataset1 import get_dataloaders, get_resampled_dataloaders, get_datasets
from utils1 import set_seed, make_deterministic

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_epoch(model, loader, optimizer, criterion, scaler):
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

def validate(model, loader, criterion):
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


SEED = 42

def execution_loop(train_batch, val_batch,
                   root, img_size, mean, std,
                   model, optimizer, epochs, training_fn, 
                   lr_scheduler, dataloader_fn=get_dataloaders, 
                   print_stats=True, **kwargs):

    trainset = get_datasets(root, 'train', img_size, mean, std)
    valset = get_datasets(root, 'valid', img_size, mean, std)
    
    trainloader = dataloader_fn(trainset, train_batch, shuffle = True, seed = SEED)
    valloader = dataloader_fn(valset, val_batch, shuffle = False, seed = SEED)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda')
    if print_stats:
        print(f"{'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  {'Val Loss':>8}  {'Val Acc':>7}")
        print("-" * 52)
    tr_loss_values = []
    vl_loss_values = []
    tr_batch_acc_values = []
    vl_batch_acc_values = []
    for epoch in range(epochs):
        
        tr_loss, tr_acc = training_fn(model, trainloader, optimizer, criterion, scaler, **kwargs)
        tr_loss_values.append(tr_loss)
        tr_batch_acc_values.append(tr_acc)

        vl_loss, vl_acc = validate(model, valloader, criterion)
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

def run_inference(root, img_size, mean, std, model_name, num_epochs, start_lr, end_lr, dataloader_fn, print_stats):
	model = create_model(model_name, pretrained=True, num_classes=6)
	model.to(device)
	optim = torch.optim.Adam(model.parameters(), lr=start_lr)
	scheduler = lr_scheduler.CosineAnnealingLR(optim, T_max=num_epochs, eta_min=end_lr)
	execution_loop(model, optim, num_epochs, train_epoch, scheduler, dataloader_fn, print_stats=print_stats)


def main():
    root = root = "./ddr/DDR-dataset/DR_grading"
    img_size = 224
    mean = (0.4140, 0.2575, 0.1289)
    std = (0.2945, 0.2047, 0.1401)
    run_inference(64, 128,
                  root, img_size, mean, std,
                  'resnet18', 50, 1e-4, 1e-7, 
                  get_resampled_dataloaders, 
                  print_stats = True)
    
if __name__ == "__main__":
    set_seed(SEED)
    main()
