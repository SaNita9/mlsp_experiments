from dotenv import load_dotenv
import os
import lightning as L
import torch
from torchvision.transforms import v2
from timm import create_model
from arg_utils import get_parser, load_yaml_config, merge_config_into_args
import tqdm
from dataset_utils import get_dataloader, get_labeled_dataset, get_resampled_dataloader, get_simclr_dataset
from simclr_pretraining import SimCLR
from model_utils import Classifier
from torch.utils.data import TensorDataset
from lightning.pytorch.loggers import CSVLogger
import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
SEED=42
NUM_WORKERS = 4
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def prepare_features_logistic_reg(network, dataset, batch_size, seed):
    network.eval()
    network.to(device)
    data_loader = get_dataloader(dataset, batch_size, shuffle=False, seed=seed)
    feats, labels = [], []
    with torch.no_grad():     
        for batch_imgs, batch_labels in tqdm.tqdm(data_loader):
            batch_imgs = batch_imgs.to(device)
            batch_feats = network(batch_imgs)
            feats.append(batch_feats.detach().cpu())
            labels.append(batch_labels)
    
    feats = torch.cat(feats)
    labels = torch.cat(labels)

    return TensorDataset(feats, labels)


def load_simclr_backbone(checkpoint_path, model_name):
    """Rebuild the timm backbone and load only its weights from a SimCLR checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]

    # Find the backbone's first conv, ignoring the projection head
    suffix = "conv1.weight"
    candidates = [k for k in state_dict
                  if k.endswith(suffix) and "projection" not in k]
    if not candidates:
        raise KeyError(f"No '{suffix}' key found. Example keys: {list(state_dict)[:10]}")

    # Shortest match is the top-level conv1, e.g. "convnet.model." (not "layer1.0.conv1.weight")
    prefix = min(candidates, key=len)[:-len(suffix)]
    print(f"Using backbone prefix: '{prefix}'")

    backbone_state = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}

    backbone = create_model(model_name, pretrained=False, num_classes=0)
    backbone.load_state_dict(backbone_state, strict=True)
    return backbone

def main():
    parser = get_parser()
    args = parser.parse_args()

    if args.config is not None:
        config_dict = load_yaml_config(args.config)
        args = merge_config_into_args(args, config_dict, parser)

    """
        nimic + conv + nimic
        nimic + self sup contrastive learning 1) 2) 3) + nimic
        nimic + sup contrastive learning + nimic
    """

    """
        imagenet + conv + nimic
        imagenet + self sup contrastive learning + nimic
        imagenet + sup contrastive learning + nimic
    """

    """
        domian-specific/other + conv + nimic
        domian-specific/other + self sup contrastive learning + nimic
        domian-specific/other + sup contrastive learning + nimic
    """

    """
        self sup contrastive learning 1) 2) 3) + conv
        sup contrastive learning + conv
    """

    # save_dir = os.path.dirname(args.checkpoint_path)
    # file_name = os.path.splitext(os.path.basename(args.checkpoint_path))[0]

    checkpoint_cb = ModelCheckpoint(
        filename="best",                    # always best.ckpt
        save_weights_only=True,
        save_top_k=1,
        mode="min",
        monitor="val_loss",
        enable_version_counter=False,       # overwrite instead of best-v1.ckpt
    )

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=10, verbose=False, mode="min")
    
    logger = CSVLogger(args.output_dir, name=args.run_name, version=args.version)
    
    trainer = L.Trainer(logger=logger,
                        accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                        devices=1,
                        max_epochs=args.num_epochs,
                        callbacks=[checkpoint_cb, early_stop_callback,
                                LearningRateMonitor("epoch")],
                        enable_progress_bar=True,
                        check_val_every_n_epoch=1,
                        deterministic=True,)

    train_data = get_labeled_dataset(args.root, "train", args.img_size, args.mean, args.std, args.clip_limit, args.tile_grid_size)
    val_data = get_labeled_dataset(args.root, "valid", args.img_size, args.mean, args.std, args.clip_limit, args.tile_grid_size)
    test_data = get_labeled_dataset(args.root, "test", args.img_size, args.mean, args.std, args.clip_limit, args.tile_grid_size)

    
    match args.pretraining_type:
        case "none": #VANILLA RESNET
            base_network = create_model(
                            args.model_name, 
                            pretrained=args.imagenet_weights,
                            num_classes=0
                        )
            resampled = args.resample
            feature_dim = None
            freeze_backbone = False

        case "SimCLR":
            # simclr_model = backbone and projection head
            # simclr_model = SimCLR.load_from_checkpoint(args.checkpoint_path)
            backbone = load_simclr_backbone(args.checkpoint_path, args.model_name)
            
            # backbone = deepcopy(simclr_model.convnet)
            # backbone.fc = nn.Identity()

            
            if args.embeddings_only:
                train_data = prepare_features_logistic_reg(backbone, dataset = train_data, batch_size = args.train_batch, seed = SEED)
                val_data = prepare_features_logistic_reg(backbone, dataset =val_data, batch_size = args.val_batch, seed = SEED)
                test_data = prepare_features_logistic_reg(backbone, dataset =test_data, batch_size = args.test_batch, seed = SEED)
                backbone.cpu()
                # base_network = nn.Linear(feature_dim=train_data.tensors[0].shape[1], num_classes=6)
                base_network = nn.Identity()
                feature_dim = train_data.tensors[0].shape[1]
                resampled = False
                freeze_backbone = True

            else: 
                base_network = backbone
                resampled = args.resample
                feature_dim = None
                freeze_backbone = False

        case _:
            raise ValueError(f"Unknown pretraining_type: {args.pretraining_type}")

    print(len(train_data), len(val_data), len(test_data)) 

    if resampled:
        train_loader = get_resampled_dataloader(train_data, args.train_batch, shuffle=False, seed = SEED)
    else:
        train_loader = get_dataloader(train_data, args.train_batch, shuffle = True, seed = SEED)
    
    val_loader = get_dataloader(val_data, args.val_batch, shuffle = False, seed = SEED)
    test_loader = get_dataloader(test_data, args.test_batch, shuffle = False, seed = SEED)


    backbone_lr_mult = 1.0 if args.pretraining_type == "none" else 0.1
    model = Classifier(base_network, 
                       feature_dim= feature_dim,
                    num_classes=6,
                    weight_decay=1e-4,
                    betas = (0.9, 0.999),
                    start_lr = args.start_lr,
                    end_lr = args.end_lr,
                    eps = args.eps,
                    warmup_epochs=10, 
                     backbone_lr_mult=backbone_lr_mult, freeze_backbone=freeze_backbone,
                     max_epochs=args.num_epochs, label_smoothing=0.0
                    )

    trainer.fit(model, train_loader, val_loader)
    # 1. Construct the path to the metrics.csv generated by CSVLogger
# Replace these variables with your actual strings if not running in the same script
    
   
    metrics_path = "/home/sara/distilaimedical/configs/the_logs/first_try/version_0/metrics.csv"
    # # 2. Read the CSV file
    # metrics_path = os.path.join("/home/sara/distilaimedical/", args.output_dir, args.run_name, "metrics.csv")
    metrics = pd.read_csv(metrics_path)
    print(metrics.columns.tolist())
        
    
        

    # 3. Align train and val metrics
    # PyTorch Lightning often logs train and val metrics on different steps/rows.
    # Grouping by epoch and taking the mean aligns them cleanly for plotting.
    metrics_by_epoch = metrics.groupby("epoch").mean().reset_index()

    # 4. Plot Loss and Accuracy
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(metrics_by_epoch["epoch"], metrics_by_epoch["train_loss_epoch"], label="Train Loss", marker="o")
    axes[0].plot(metrics_by_epoch["epoch"], metrics_by_epoch["val_loss"], label="Val Loss", marker="o")
    axes[0].set_title("Training and Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    # --- Plot Accuracy ---
    axes[1].plot(metrics_by_epoch["epoch"], metrics_by_epoch["train_acc_epoch"], label="Train Accuracy", marker="o")
    axes[1].plot(metrics_by_epoch["epoch"], metrics_by_epoch["val_acc"], label="Val Accuracy", marker="o")
    axes[1].set_title("Training and Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()
    trainer.test(model, test_loader, ckpt_path="best")  
    print(trainer.checkpoint_callback.best_model_path)

if __name__ == "__main__":
    
    L.seed_everything(SEED, workers=True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.set_float32_matmul_precision('medium')
    load_dotenv()
    main()