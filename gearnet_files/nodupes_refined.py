import os
import torch
from torchdrug import models, transforms, datasets, data, layers, tasks, core
from torchdrug.layers import geometry
from sklearn.model_selection import StratifiedKFold
from torch.utils import data as torch_data
import pandas as pd
import numpy as np
import random
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
import time
import wandb
from sklearn.metrics import confusion_matrix, classification_report

# Set all random seeds for reproducibility
def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Configuration
class Config:
    seed = 56
    batch_size = 35
    num_epochs = 2500
    learning_rate = 1e-5 ## 1e-6
    weight_decay = 0.01
    max_lr = 1e-4
    gradient_clip = 1.0
    patience = 65
    delta = 0.001
    num_folds = 5
    
    # Paths
    pdb_folder = '/work/hdd/bdja/bpokhrel/gearnet_files/pdb_m15_ac'
    soluble_folder_ac = '/work/hdd/bdja/bpokhrel/gearnet_files/watersoluble_proteins_ac'
    csv_path = '/work/hdd/bdja/bpokhrel/gearnet_files/final_pdb_csv2.csv'
    output_dir = '/u/bpokhrel/gearnet_files/nodupes_refined/'
    
    # Model architecture
    hidden_dims = [512, 512, 512]
    num_mlp_layers = 4
    mlp_dropout = 0.2
    
    # WandB
    project_name = "nodupes_trim_sol3_refined"
    
set_seeds(Config.seed)

# Enhanced Protein Dataset Loader
class ProteinLoader(data.ProteinDataset):
    def __init__(self, csv_path, transform, random_seed=None, balance=True):
        self.pdb_folder = Config.pdb_folder
        self.soluble_folder_ac = Config.soluble_folder_ac
        self.balance = balance
        self.random_seed = random_seed
        
        self.csv_path = csv_path
        self.transform = transform
        self.label_dict, self.df, self.k, self.label_population = self._prepare_data()
        self.pdb_list = self.df['pdbid'].tolist()
        self.pdbs = self.df['pdbid']
        
        self.data, self.missing_pdbs = self._load_proteins()
        self.label_list = self._get_labels()
        self.targets = {'label': self.label_list}
        self.num_samples = len(self.data)
        
    def _prepare_data(self):
        """Load and prepare the dataset with balanced classes"""
        df = pd.read_csv(self.csv_path)
        df = df[['pdbid', 'membrane_name_cache']]
        print(f'Initial dataset shape: {df.shape}')
        
        # Add soluble proteins
        soluble_pdbs = [f for f in os.listdir(self.soluble_folder_ac) if not f.startswith('.')]
        selected_pdbs = random.sample(soluble_pdbs, min(500, len(soluble_pdbs)))
        new_rows = [{'pdbid': os.path.splitext(s_pdb)[0], 'membrane_name_cache': 'z_soluble'} 
                   for s_pdb in selected_pdbs]
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        print(f'Final dataset shape: {df.shape}')

        # Balance classes if requested
        if self.balance:
            def select_rows(group):
                return group.sample(n=100) if len(group) >= 100 else group
            df = df.groupby('membrane_name_cache', group_keys=False).apply(select_rows).reset_index(drop=True)
        
        # Create label dictionary
        counts = df['membrane_name_cache'].value_counts()
        print('Class distribution:\n', counts)
        
        location = df['membrane_name_cache'].unique()
        label_dict = {key: value for value, key in enumerate(sorted(location))}
        
        return label_dict, df, len(label_dict), counts
    
    def _load_proteins(self):
        """Load protein structures from PDB files"""
        protein_list = []
        missing_pdbs = []
        
        for pdb in self.pdbs:
            try:
                if pdb.startswith('.'): 
                    continue
                    
                pdb_path = os.path.join(
                    self.soluble_folder_ac if pdb.startswith('pdb') else self.pdb_folder, 
                    pdb
                )
                
                # Load protein with enhanced features
                iprotein = data.Protein.from_pdb(
                    pdb_path, 
                    atom_feature="position", 
                    bond_feature="length", 
                    residue_feature="symbol"
                )
                
                # Graph construction with multiple edge types
                filter_alpha = layers.GraphConstruction(
                    node_layers=[geometry.AlphaCarbonNode()],
                    edge_layers=[
                        geometry.SpatialEdge(radius=10.0, min_distance=5),
                        geometry.KNNEdge(k=10, min_distance=5),
                        geometry.SequentialEdge(max_distance=2)
                    ],
                    edge_feature="gearnet"
                )
                
                protein = filter_alpha(iprotein)
                if protein is None:
                    missing_pdbs.append(pdb)
                    continue
                    
                protein_list.append(protein)
                
            except Exception as e:
                print(f"Error loading {pdb}: {str(e)}")
                missing_pdbs.append(pdb)
                
        print(f"Could not load {len(missing_pdbs)} proteins")
        return protein_list, missing_pdbs
    
    def _get_labels(self):
        """Get labels for successfully loaded proteins"""
        label_list = []
        for pdb in self.pdbs:
            if pdb not in self.missing_pdbs:
                label = self.df.loc[self.df['pdbid'] == pdb, 'membrane_name_cache'].values[0]
                label_list.append(self.label_dict[label])
        return label_list
    
    def stratified_k_fold_split(self, n_splits=5):
        """Create stratified k-fold splits"""
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_seed)
        labels = self.targets["label"]
        folds = []
        
        for train_idx, val_idx in skf.split(range(len(self.data)), labels):
            # Print class distribution for each fold
            train_labels = [labels[i] for i in train_idx]
            val_labels = [labels[i] for i in val_idx]
            
            print(f"\nFold {len(folds)+1} distribution:")
            print("Train:", pd.Series(train_labels).value_counts().sort_index().to_dict())
            print("Valid:", pd.Series(val_labels).value_counts().sort_index().to_dict())
            
            folds.append((
                torch_data.Subset(self, train_idx),
                torch_data.Subset(self, val_idx)
            ))
            
        return folds
    
    def __getitem__(self, index):
        protein = self.data[index]
        label = self.label_list[index]
        name = self.pdb_name[index]
        item = {'graph': protein, 'label': label, 'name': name}
        return self.transform(item) if self.transform else item
    
    def __len__(self):
        return len(self.data)

# Enhanced Early Stopping
class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.best_metrics = None

    def __call__(self, val_loss, val_metrics=None):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.best_metrics = val_metrics
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_metrics = val_metrics
            self.counter = 0

# Model Initialization
def initialize_model(num_classes):
    # Enhanced GearNet with more features
    gearnet = models.GearNet(
        input_dim=21,
        hidden_dims=Config.hidden_dims,
        num_relation=7,
        batch_norm=True,
        concat_hidden=True,  # Use all layer outputs
        short_cut=True,
        readout="sum"  # Try attention readout
    )
    
    # Enhanced graph construction
    graph_construction = layers.GraphConstruction(
        node_layers=[geometry.AlphaCarbonNode()],
        edge_layers=[
            geometry.SpatialEdge(radius=10.0, min_distance=5),
            geometry.KNNEdge(k=10, min_distance=5),
            geometry.SequentialEdge(max_distance=2)
        ],
        edge_feature="gearnet"
    )
    
    # Enhanced task with more metrics
    task = tasks.PropertyPrediction(
        gearnet,
        graph_construction_model=graph_construction,
        num_mlp_layer=Config.num_mlp_layers,
        num_class=num_classes,
        mlp_batch_norm=True,
        mlp_dropout=Config.mlp_dropout,
        task=['label'],
        criterion="ce",
        metric=["acc", "auroc", "f1"]
    )
    
    return task

# Evaluation Function
def evaluate_model(solver, dataset, prefix="valid"):
    metrics = solver.evaluate(dataset)
    predictions = solver.predict(dataset)
    
    # Get true labels
    true_labels = []
    for item in dataset:
        true_labels.append(item['label'])
    true_labels = torch.tensor(true_labels)
    
    # Get predicted labels
    pred_labels = predictions['label'].argmax(dim=1)
    
    # Calculate additional metrics
    cm = confusion_matrix(true_labels, pred_labels)
    cr = classification_report(true_labels, pred_labels, output_dict=True)
    
    # Log to WandB
    wandb.log({
        f"{prefix}/confusion_matrix": wandb.plot.confusion_matrix(
            preds=pred_labels.numpy(),
            y_true=true_labels.numpy(),
            class_names=list(dataset.dataset.label_dict.keys())
        ),
        f"{prefix}/precision": cr['weighted avg']['precision'],
        f"{prefix}/recall": cr['weighted avg']['recall'],
        f"{prefix}/f1": cr['weighted avg']['f1-score']
    })
    
    return metrics

# Main Training Function
def train_fold(fold_idx, train_set, valid_set, num_classes):
    # Initialize WandB
    wandb.init(
        project=Config.project_name,
        name=f"fold_{fold_idx}",
        config={
            "learning_rate": Config.learning_rate,
            "batch_size": Config.batch_size,
            "architecture": "GearNet",
            "dataset": "ProteinMembraneClassification",
            "epochs": Config.num_epochs
        }
    )
    
    # Initialize model
    model = initialize_model(num_classes)
    wandb.watch(model, log='all', log_freq=5)
    
    # Enhanced optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.learning_rate,
        weight_decay=Config.weight_decay
    )
    
    # Enhanced learning rate scheduler
    scheduler = OneCycleLR(
        optimizer,
        max_lr=Config.max_lr,
        steps_per_epoch=len(train_set)//Config.batch_size,
        epochs=Config.num_epochs
    )
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=Config.patience,
        verbose=True,
        delta=Config.delta
    )
    
    # Output paths
    best_model_path = os.path.join(
        Config.output_dir,
        f"best_model_fold_{fold_idx}.pth"
    )
    final_model_path = os.path.join(
        Config.output_dir,
        f"final_model_fold_{fold_idx}.pth"
    )
    
    # Create solver
    solver = core.Engine(
        model,
        train_set,
        valid_set,
        None,
        optimizer,
        batch_size=Config.batch_size,
        gpus=[0],
        gradient_clip=Config.gradient_clip,
        best_model_path=best_model_path,
        early_stopping=early_stopping,
        scheduler=scheduler
    )
    
    # Training loop
    for epoch in range(Config.num_epochs):
        # Train step
        train_metrics = solver.train_step()
        
        # Validation step
        valid_metrics = evaluate_model(solver, valid_set)
        
        # Update learning rate
        scheduler.step()
        
        # Log metrics
        wandb.log({
            "train/loss": train_metrics['loss'],
            "train/accuracy": train_metrics['acc'],
            "valid/loss": valid_metrics['loss'],
            "valid/accuracy": valid_metrics['acc'],
            "valid/auroc": valid_metrics['auroc'],
            "valid/f1": valid_metrics['f1'],
            "lr": scheduler.get_last_lr()[0]
        })
        
        # Early stopping check
        early_stopping(valid_metrics['loss'], valid_metrics)
        
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch}")
            break
    
    # Save final model
    solver.save(final_model_path)
    
    # Final evaluation
    final_metrics = evaluate_model(solver, valid_set)
    print(f"\nFinal metrics for fold {fold_idx}:")
    print(final_metrics)
    
    wandb.finish()
    
    return final_metrics

# Main Execution
if __name__ == "__main__":
    # Create output directory if it doesn't exist
    os.makedirs(Config.output_dir, exist_ok=True)
    
    # Load dataset
    transform = transforms.ProteinView(view='residue')
    dataset = ProteinLoader(
        csv_path=Config.csv_path,
        transform=transform,
        random_seed=Config.seed,
        balance=False
    )
    
    # Get number of classes
    num_classes = dataset.k
    print(f"Number of classes: {num_classes}")
    
    # Create stratified folds
    folds = dataset.stratified_k_fold_split(n_splits=Config.num_folds)
    
    # Train each fold
    all_metrics = []
    for fold_idx, (train_set, valid_set) in enumerate(folds):
        print(f"\n{'='*40}")
        print(f"Training fold {fold_idx+1}/{Config.num_folds}")
        print(f"{'='*40}\n")
        
        metrics = train_fold(fold_idx+1, train_set, valid_set, num_classes)
        all_metrics.append(metrics)
    
    # Print summary of all folds
    print("\nTraining complete. Summary of all folds:")
    for i, metrics in enumerate(all_metrics, 1):
        print(f"Fold {i}:")
        print(f"  Accuracy: {metrics['acc']:.4f}")
        print(f"  AUROC: {metrics['auroc']:.4f}")
        print(f"  F1: {metrics['f1']:.4f}")
        print(f"  Loss: {metrics['loss']:.4f}")
