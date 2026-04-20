import os
import torch
import random
import numpy as np
import argparse
import wandb
import torch.optim.lr_scheduler as lr_scheduler
import traceback
from torchdrug import core

# --- Import from your custom module ---
from gearnet_modules import ProteinLoader, initialize_model, EarlyStopping, transform

def parse_arguments():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description='Final Protein Classification with GearNet')
    parser.add_argument('--pdb_folder', type=str, required=True, help='Path to PDB folder containing membrane proteins')
    parser.add_argument('--soluble_folder_ac', type=str, required=True, help='Path to soluble proteins folder')
    
    # NEW: Replaced single csv_path with split_dir
    parser.add_argument('--split_dir', type=str, required=True, help='Path to the Inner Fold directory (e.g., splits/Outer_Fold_1/Inner_Fold_1)')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for models and results')
    parser.add_argument('--num_epochs', type=int, default=2500, help='Number of training epochs')
    
    
    # Randomness
    parser.add_argument('--seed', type=int, default=56, help='Random seed for reproducibility')
    # --- HYPERPARAMETERS ---
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--mlp_dropout', type=float, default=0.2)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--num_gearnet_layers', type=int, default=3)
    parser.add_argument('--readout', type=str, default='sum')
    parser.add_argument('--concat_hidden', type=str, default='True') 
    parser.add_argument('--knn_k', type=int, default=10)
    parser.add_argument('--spatial_radius', type=float, default=10.0)
    parser.add_argument('--activation', type=str, default='relu', help='Activation function for GearNet (e.g., relu, silu, gelu)') # <--- ADDED HERE

    
    return parser.parse_args()

class Config:
    def __init__(self, args):
        self.pdb_folder = args.pdb_folder
        self.soluble_folder_ac = args.soluble_folder_ac
        self.split_dir = args.split_dir
        self.output_dir = args.output_dir
        self.num_epochs = args.num_epochs
        self.seed = args.seed
        
        self.batch_size = args.batch_size
        self.hidden_dim = args.hidden_dim
        self.learning_rate = args.learning_rate
        self.mlp_dropout = args.mlp_dropout
        self.weight_decay = args.weight_decay
        self.num_gearnet_layers = args.num_gearnet_layers
        self.readout = args.readout
        self.concat_hidden = str(args.concat_hidden).lower() == 'true'
        self.knn_k = args.knn_k
        self.spatial_radius = args.spatial_radius
        self.activation = args.activation

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    args = parse_arguments()
    config = Config(args)
    set_seeds(config.seed)
    
    os.makedirs(config.output_dir, exist_ok=True)
    
    # --- 1. RESOLVE MANIFEST PATHS ---
    train_csv = os.path.join(config.split_dir, "train_manifest.csv")
    valid_csv = os.path.join(config.split_dir, "valid_manifest.csv")
    # Test set is located one directory up (in the Outer_Fold folder)
    outer_fold_dir = os.path.dirname(config.split_dir)
    test_csv = os.path.join(outer_fold_dir, "test_manifest.csv")
    
    # Sanity check to ensure files exist before starting
    for path in [train_csv, valid_csv, test_csv]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required split file: {path}")

    # --- 2. INITIALIZE WANDB DYNAMICALLY ---
    outer_name = os.path.basename(outer_fold_dir)
    inner_name = os.path.basename(config.split_dir)
    run_name = f"{outer_name}_{inner_name}"
    
    wandb.init(
        project="train_productionv2", 
        name=run_name, 
        config=vars(args)
    )
    
    # --- 3. LOAD DATASETS FROM SPECIFIC MANIFESTS ---
    print(f"Loading datasets for {run_name}...")
    
    train_set = ProteinLoader(
        csv_path=train_csv, pdb_folder=config.pdb_folder,
        soluble_folder_ac=config.soluble_folder_ac, transform=transform
    )
    valid_set = ProteinLoader(
        csv_path=valid_csv, pdb_folder=config.pdb_folder,
        soluble_folder_ac=config.soluble_folder_ac, transform=transform
    )
    test_set = ProteinLoader(
        csv_path=test_csv, pdb_folder=config.pdb_folder,
        soluble_folder_ac=config.soluble_folder_ac, transform=transform
    )

    # Calculate weights STRICTLY from the training set
    num_classes = train_set.k 
    wt = train_set.wt
    
    print(f"Data loaded successfully.")
    print(f"Training samples:   {len(train_set)}")
    print(f"Validation samples: {len(valid_set)}")
    print(f"Test samples:       {len(test_set)}")
    print(f"Number of classes:  {num_classes}")

    # --- 4. INITIALIZE MODEL & ENGINE ---
    # Save the model with the fold name so they don't overwrite each other
    best_model_path = os.path.join(config.output_dir, f'{run_name}_best_model.pth')
    
    task = initialize_model(
        num_classes, class_weights=wt, mlp_dropout=config.mlp_dropout, 
        hidden_dim=config.hidden_dim, num_gearnet_layers=config.num_gearnet_layers,
        readout=config.readout, concat_hidden=config.concat_hidden,
        knn_k=config.knn_k, spatial_radius=config.spatial_radius, activation=config.activation
    )
    
    optimizer = torch.optim.AdamW(task.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    early_stopping = EarlyStopping(patience=15, verbose=True)
    
    solver = core.Engine(
        task=task,
        train_set=train_set,
        valid_set=valid_set,
        test_set=test_set, 
        optimizer=optimizer,
        scheduler=scheduler,
        early_stopping=early_stopping,
        best_model_path=best_model_path, 
        batch_size=config.batch_size,
        gpus=[0]
    )
    
    # --- 5. TRAIN & EVALUATE ---
    print(f"\nStarting training for {run_name}...")
    solver.train(num_epoch=config.num_epochs)
    last_model_path = os.path.join(config.output_dir, f'{run_name}_last_epoch_model.pth')
    solver.save(last_model_path)
    
    print("\nLoading the best weights from training...")
    solver.load(best_model_path) 
    # solver.load(last_model_path) 
    
    
    print("\nEvaluating model on the completely unseen TEST set...")
    try:
        test_metrics = solver.evaluate("test")
        print(f"\n=== FINAL TEST METRICS FOR {run_name} ===")
        
        for key, val in test_metrics.items():
            print(f"{key}: {val:.4f}")
            
        wandb_log_dict = {f"test_{key.replace(' ', '_')}": float(val) for key, val in test_metrics.items()}
        wandb.log(wandb_log_dict)
        
    except Exception as e:
        print(f"Test evaluation failed: {e}")
        traceback.print_exc()
        
    wandb.finish()
    print("\nRun complete. Model saved at:", best_model_path)