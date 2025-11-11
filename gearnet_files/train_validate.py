import os
import torch
from torchdrug import models, transforms, datasets, data, layers, tasks, core
from torchdrug.layers import geometry
from sklearn.model_selection import StratifiedKFold
from torch.utils import data as torch_data
import pandas as pd
import numpy as np
import random
import torch.optim.lr_scheduler as lr_scheduler
import wandb
import argparse

# Command line arguments
def parse_arguments():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description='Protein Classification with GearNet')
    parser.add_argument('--pdb_folder', type=str, required=True, 
                        help='Path to PDB folder containing membrane proteins')
    parser.add_argument('--soluble_folder_ac', type=str, required=True,
                        help='Path to soluble proteins folder')
    parser.add_argument('--csv_path', type=str, required=True,
                        help='Path to CSV file with protein metadata')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for models and results')
    parser.add_argument('--num_epochs', type=int, default=2500,
                        help='Number of training epochs')
    
    # Use action='store_true' for boolean flags. 
    # This means adding '--training' to the command will set it to True.
    # If the flag is absent, it will be False by default.
    parser.add_argument('--training', action='store_true',
                        help='Flag to run in training mode. If absent, runs in evaluation mode.')
    
    parser.add_argument('--model_path_folder', type=str, default=None,
                        help='Path to pre-trained model weights (required if not in training mode)')
    parser.add_argument('--best_or_last', type=str, default='last',
                         choices=['best', 'last'],
                         help='Load either the best or last epoch (default: last)')
    return parser.parse_args()

# Configuration
class Config:
    """Stores configuration parameters."""
    def __init__(self, args):
        self.pdb_folder = args.pdb_folder
        self.soluble_folder_ac = args.soluble_folder_ac
        self.csv_path = args.csv_path
        self.output_dir = args.output_dir
        self.num_epochs = args.num_epochs
        self.training = args.training
        self.model_path_folder = args.model_path_folder
        self.best_or_last = args.best_or_last
        self.seed = 56 # Hardcoded seed for reproducibility

# Set random seed for reproducibility
def set_seeds(seed):
    """Sets random seeds for all relevant libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set the global seed
SEED = 56
set_seeds(SEED)

transform = transforms.ProteinView(view='residue')

class ProteinLoader(data.ProteinDataset):
    """Custom dataset loader for protein data with membrane classification."""
    
    def __init__(self, csv_path: str, pdb_folder: str, soluble_folder_ac: str, transform, random_seed: int = None, balance: bool = True):
        """
        Initialize the ProteinLoader.
        
        Args:
            csv_path: Path to CSV file containing protein data
            pdb_folder: Path to PDB folder
            soluble_folder_ac: Path to soluble proteins folder
            transform: Transformation to apply to proteins
            random_seed: Random seed for reproducibility
            balance: Whether to balance the dataset
        """
        self.pdb_folder = pdb_folder
        self.soluble_folder_ac = soluble_folder_ac
        self.balance = balance
        self.random_seed = random_seed
        
        if random_seed is not None:
            # This seeds np for sampling if balance=True
            np.random.seed(random_seed)
            # The global `random.seed(SEED)` call outside will control `random.sample`
            
        self.csv_path = csv_path
        self.transform = transform
        self.label_dict, self.df, self.k, self.label_population = self._make_dict()
        self.pdb_list = self.df['pdbid'].tolist()
        self.pdbs = self.df['pdbid']
        
        # Load proteins in the order determined by the dataframe
        self.data, self.pdb_list2, self.pdb_list3, self.pdb_name = self._load_proteins()
        self.label_list = self._get_labels()
        self.targets = {'label': self.label_list}
        self.num_samples = len(self.data)
        
    def _make_dict(self):
        """Create label dictionary and process the input dataframe."""
        try:
            df = pd.read_csv(self.csv_path)
            df = df[['pdbid', 'membrane_name_cache']]
            print(f'Initial dataframe shape: {df.shape}')
            
            # Add soluble proteins
            soluble_pdbs = [f for f in os.listdir(self.soluble_folder_ac) if not f.startswith('.')]
            
            # --- CRITICAL FIX FOR REPRODUCIBILITY ---
            # We MUST sort the list of filenames before sampling.
            # os.listdir() does not guarantee the same file order across runs.
            # Sorting ensures that random.sample() picks the exact same files
            # every time, given that the global random.seed(56) is set.
            soluble_pdbs.sort()
            
            # This sample uses the global `random` state, seeded to 56 by set_seeds()
            selected_pdbs = random.sample(soluble_pdbs, min(500, len(soluble_pdbs)))
            
            new_rows = [{'pdbid': os.path.splitext(s_pdb)[0], 'membrane_name_cache': 'z_soluble'} 
                        for s_pdb in selected_pdbs]
            
            # Concatenating maintains order: original df first, then new_rows
            df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
            print(f'Final dataframe shape: {df.shape}')
            
            # Calculate label distribution
            counts = df['membrane_name_cache'].value_counts()
            label_dict_counts = {key: counts[key] for key in counts.index}
            print('Label distribution:', label_dict_counts)
            
            location = df['membrane_name_cache'].unique()
            label_dict = {key: value for value, key in enumerate(sorted(location))}
            
            if self.balance:
                # Note: .sample() here uses np.random, which was seeded by self.random_seed
                def select_100_rows(group):
                    # Added random_state to group.sample for reproducibility if balancing
                    return group.sample(n=100, random_state=self.random_seed) if len(group) >= 100 else group
                df = df.groupby('membrane_name_cache', group_keys=False).apply(select_100_rows).reset_index(drop=True)
                
            return label_dict, df, len(label_dict), label_dict_counts
            
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            raise
            
    def _load_proteins(self):
        """Load protein structures from PDB files."""
        protein_list = []
        pdb_name = []
        pdb_list2 = []  # Failed to load
        pdb_list3 = []  # File not found
        
        # Iterates in the exact order of the dataframe (self.pdbs)
        for pdb in self.pdbs:
            try:
                if pdb.startswith('.'):
                    continue
                    
                pdb_path = os.path.join(
                    self.soluble_folder_ac if pdb.startswith('pdb') else self.pdb_folder, 
                    pdb
                )
                
                iprotein = data.Protein.from_pdb(
                    pdb_path, 
                    atom_feature="position", 
                    bond_feature="length", 
                    residue_feature="symbol"
                )
                
                filter_alpha = layers.GraphConstruction(node_layers=[geometry.AlphaCarbonNode()])
                protein = filter_alpha(iprotein)
                
                if protein is None:
                    pdb_list2.append(pdb)
                    continue
                    
                protein_list.append(protein)
                pdb_name.append(pdb)
                
            except FileNotFoundError:
                pdb_list3.append(pdb)
            except Exception as e:
                print(f"Error loading protein {pdb}: {e}")
                pdb_list2.append(pdb)
                
        print(f"Successfully loaded {len(protein_list)} proteins")
        print(f"Failed to load {len(pdb_list2)} proteins")
        print(f"File not found for {len(pdb_list3)} proteins")
        
        return protein_list, pdb_list2, pdb_list3, pdb_name
        
    def _get_labels(self):
        """Get labels for successfully loaded proteins, maintaining order."""
        return [
            self.label_dict[self.df.loc[self.df['pdbid'] == pdb, 'membrane_name_cache'].values[0]]
            for pdb in self.pdbs 
            if pdb not in self.pdb_list2 and pdb not in self.pdb_list3
        ]
        
    def stratified_k_fold_split(self, n_splits: int = 5):
        """Generate stratified k-fold splits of the dataset."""
        # This uses the seed (56) passed during __init__
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_seed)
        val_idx = []
        folds = []
        labels = self.targets["label"]

        # skf.split will produce identical indices because:
        # 1. The data (range(len(self.data))) is the same length.
        # 2. The labels are in the same order.
        # 3. random_state is the same (56).
        for train_indices, val_indices in skf.split(range(len(self.data)), labels):
            train_subset = torch_data.Subset(self, train_indices)
            val_subset = torch_data.Subset(self, val_indices)
            folds.append((train_subset, val_subset))
            val_idx.append(val_indices)
            
        # Print label distribution for each fold for verification
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(range(len(self.data)), labels)):
            train_labels = [labels[i] for i in train_idx]
            val_labels = [labels[i] for i in val_idx]
            print(f"Fold {fold_idx + 1}:")
            print("  Training label distribution:", pd.Series(train_labels).value_counts().sort_index().to_dict())
            print("  Validation label distribution:", pd.Series(val_labels).value_counts().sort_index().to_dict())
            
        return folds, val_idx
        
    def __getitem__(self, index):
        protein = self.data[index]
        label = self.label_list[index]
        name = self.pdb_name[index]
        item = {'graph': protein, 'label': label, 'name': name}
        return self.transform(item) if self.transform else item
        
    def __len__(self):
        return len(self.data)

class EarlyStopping:
    """Early stops training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience: int = 10, verbose: bool = False, delta: float = 0, counter: int = 0):
        self.patience = patience
        self.verbose = verbose
        self.counter = counter
        self.best_loss = np.Inf
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss):
        if self.best_loss == np.Inf:
            self.best_loss = val_loss
        elif val_loss > self.best_loss + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def initialize_model(num_classes: int):
    """Initialize the GearNet model with fixed initialization."""
    # Set seed again right before model init for reproducible weights
    torch.manual_seed(56) 
    
    gearnet_edge = models.GearNet(
        input_dim=21, 
        hidden_dims=[512, 512, 512], 
        num_relation=7,
        batch_norm=True, 
        concat_hidden=False, 
        short_cut=True, 
        readout="sum"
    )
    
    graph_construction_model = layers.GraphConstruction(
        node_layers=[geometry.AlphaCarbonNode()], 
        edge_layers=[
            geometry.SpatialEdge(radius=10.0, min_distance=5),
            geometry.KNNEdge(k=10, min_distance=5),
            geometry.SequentialEdge(max_distance=2)
        ],
        edge_feature=None
    )

    task = tasks.PropertyPrediction(
        gearnet_edge, 
        graph_construction_model=graph_construction_model, 
        num_mlp_layer=4, 
        num_class=num_classes,
        mlp_batch_norm=False, 
        mlp_dropout=0.2,
        task=['label'], 
        criterion="ce", 
        metric=["acc"]
    )
    
    return task

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Initialize configuration
    config = Config(args)
    
    # Set random seeds (already done, but good practice in main)
    set_seeds(config.seed)
    
    # Set W&B for reproducibility
    os.environ['WANDB_RUN_ID'] = str(config.seed)
    os.environ['WANDB_RESUME'] = 'allow'
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Load dataset
    print("Loading dataset...")
    dataset = ProteinLoader(
        csv_path=config.csv_path,
        pdb_folder=config.pdb_folder,
        soluble_folder_ac=config.soluble_folder_ac,
        transform=transform,
        random_seed=config.seed, # This seed (56) is passed to StratifiedKFold
        balance=False # As requested, balance is set to False
    )
    num_classes = dataset.k 
    print(f"Dataset loaded. Number of classes: {num_classes}")

    # Generate folds
    print("Generating stratified k-fold splits...")
    folds, valid_indices = dataset.stratified_k_fold_split(n_splits=5)
    print(f"Generated {len(folds)} folds.")

    for fold_idx, (train_set, valid_set) in enumerate(folds, 1):
        print(f"\n--- Processing Fold {fold_idx} ---")
        
        # Initialize WandB
        wandb.init(
            project="nodupes_trim_sol3", # Make sure this project name is correct
            name=f"nodupes_trim_sol3_{fold_idx}", # And this run name
            config={
                "learning_rate": 1e-6,
                "batch_size": 35,
                "architecture": "GearNet",
                "dataset": "ProteinMembraneClassification",
                "seed": config.seed,
                "fold": fold_idx
            },
            settings=wandb.Settings(start_method="fork")
        )
        
        # Define model paths for THIS fold
        best_model_path = os.path.join(config.output_dir, f'nodupes_trim_best_sol2_{fold_idx}.pth')
        final_model_path = os.path.join(config.output_dir, f'nodupes_trim_last_sol2_{fold_idx}.pth')
        
        # Model and training setup
        task = initialize_model(num_classes)
        wandb.watch(task, log='all', log_freq=5)
        
        optimizer = torch.optim.Adam(task.parameters(), lr=1e-6)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=35)
        early_stopping = EarlyStopping(patience=65, verbose=True)
        
        # Initialize engine
        solver = core.Engine(
            task=task,
            train_set=train_set,
            valid_set=valid_set,
            test_set=None,
            optimizer=optimizer,
            scheduler=scheduler,
            early_stopping=early_stopping,
            best_model_path=best_model_path, # Path to save best model during training
            batch_size=35,
            gpus=[0]
        )
        
        # Training vs Evaluation logic
        if config.training:
            print(f"Training fold {fold_idx}...")
            solver.train(num_epoch=config.num_epochs)
            solver.save(final_model_path)
            print(f"Saved final model to {final_model_path}")
            # After training, evaluate on the validation set
            print("Evaluating best model from training...")
            solver.load(best_model_path) # Load best model
        else:
            # Load pre-trained model
            if config.model_path_folder is None:
                print("Error: `model_path_folder` is required when not training.")
                wandb.finish()
                continue

            if config.best_or_last == 'last':
                wt_path = os.path.join(
                    config.model_path_folder, 
                    f"nodupes_trim_last_sol2_{fold_idx}.pth" # Make sure this matches your file names
                )
            else: # 'best'
                wt_path = os.path.join(
                    config.model_path_folder, 
                    f"nodupes_trim_best_sol2_{fold_idx}.pth" # Make sure this matches your file names
                )
            
            print(f'📂 Loading model from {wt_path}')
            
            try: 
                solver.load(wt_path)
                print("  Model loaded successfully.")
            except Exception as e:
                print(f'  Failed to load model: {e}')
                wandb.finish()
                continue
        
        # Evaluation (for both training and loaded models)
        print(f"Evaluating model on validation set for fold {fold_idx}...")
        try:
            metrics = solver.evaluate("valid")
            
            # Handle different possible metric keys
            accuracy = metrics.get('acc', metrics.get('accuracy', float('nan')))
            val_loss = metrics.get('loss', float('nan'))
            
            print(f"\nFold {fold_idx} Validation Metrics:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Loss: {val_loss:.4f}")
            
            wandb.log({
                "final_val_accuracy": accuracy,
                "final_val_loss": val_loss,
                "fold": fold_idx
            })
            
        except Exception as e:
            print(f"Evaluation failed: {e}")
        
        # Cleanup
        wandb.finish()
        del solver
        del task
        del optimizer
        torch.cuda.empty_cache()
    
    print("\nAll folds processed.")lida