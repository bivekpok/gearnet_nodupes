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
    parser = argparse.ArgumentParser(description='Protein Classification with GearNet')
    parser.add_argument('--pdb_folder', type=str, required=True, 
                       help='Path to PDB folder containing membrane proteins')
    parser.add_argument('--soluble_folder', type=str, required=True,
                       help='Path to soluble proteins folder')
    parser.add_argument('--csv_path', type=str, required=True,
                       help='Path to CSV file with protein metadata')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for models and results')
    parser.add_argument('--num_epochs', type=int, default=2500,
                       help='Number of training epochs')
    return parser.parse_args()

# Configuration
class Config:
    def __init__(self, args):
        self.pdb_folder = args.pdb_folder
        self.soluble_folder_ac = args.soluble_folder
        self.csv_path = args.csv_path
        self.output_dir = args.output_dir
        self.num_epochs = args.num_epochs
        self.seed = 56

# Set random seed for reproducibility
def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
            np.random.seed(random_seed)
            random.seed(random_seed)
            
        self.csv_path = csv_path
        self.transform = transform
        self.label_dict, self.df, self.k, self.label_population = self._make_dict()
        self.pdb_list = self.df['pdbid'].tolist()
        self.pdbs = self.df['pdbid']
        
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
            selected_pdbs = random.sample(soluble_pdbs, min(500, len(soluble_pdbs)))
            new_rows = [{'pdbid': os.path.splitext(s_pdb)[0], 'membrane_name_cache': 'z_soluble'} 
                        for s_pdb in selected_pdbs]
            
            df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
            print(f'Final dataframe shape: {df.shape}')
            
            # Calculate label distribution
            counts = df['membrane_name_cache'].value_counts()
            label_dict_counts = {key: counts[key] for key in counts.index}
            print('Label distribution:', label_dict_counts)
            
            location = df['membrane_name_cache'].unique()
            label_dict = {key: value for value, key in enumerate(sorted(location))}
            
            if self.balance:
                def select_100_rows(group):
                    return group.sample(n=100) if len(group) >= 100 else group
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
        """Get labels for successfully loaded proteins."""
        return [
            self.label_dict[self.df.loc[self.df['pdbid'] == pdb, 'membrane_name_cache'].values[0]]
            for pdb in self.pdbs 
            if pdb not in self.pdb_list2 and pdb not in self.pdb_list3
        ]
        
    def stratified_k_fold_split(self, n_splits: int = 5):
        """Generate stratified k-fold splits of the dataset."""
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_seed)
        val_idx = []
        folds = []
        labels = self.targets["label"]

        for train_indices, val_indices in skf.split(range(len(self.data)), labels):
            train_subset = torch_data.Subset(self, train_indices)
            val_subset = torch_data.Subset(self, val_indices)
            folds.append((train_subset, val_subset))
            val_idx.append(val_indices)
            
            # Print label distribution for each fold
            train_labels = [labels[i] for i in train_indices]
            val_labels = [labels[i] for i in val_indices]
            print(f"Fold {len(folds)}:")
            print("Training label distribution:", pd.Series(train_labels).value_counts().to_dict())
            print("Validation label distribution:", pd.Series(val_labels).value_counts().to_dict())
            
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
    torch.manual_seed(56)  # For reproducible model initialization
    
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
    
    # Set random seeds
    set_seeds(config.seed)
    
    # Set W&B for reproducibility
    os.environ['WANDB_RUN_ID'] = str(config.seed)
    os.environ['WANDB_RESUME'] = 'allow'
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Load dataset
    dataset = ProteinLoader(
        csv_path=config.csv_path,
        pdb_folder=config.pdb_folder,
        soluble_folder_ac=config.soluble_folder_ac,
        transform=transform,
        random_seed=config.seed,
        balance=False
    )
    num_classes = dataset.k 

    def initialize_model():
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

    # Generate folds
    folds, valid_indices = dataset.stratified_k_fold_split(n_splits=5)

    for fold_idx, (train_set, valid_set) in enumerate(folds, 1):
        # Initialize WandB with proper settings
        wandb.init(
            project="nodupes_trim_sol3",
            name=f"nodupes_trim_sol3_{fold_idx}",
            config={
                "learning_rate": 1e-6,
                "batch_size": 35,
                "architecture": "GearNet",
                "dataset": "ProteinMembraneClassification",
                "seed": config.seed
            },
            settings=wandb.Settings(start_method="fork")
        )
        
        # Model and training setup
        task = initialize_model()
        wandb.watch(task, log='all', log_freq=5)
        
        optimizer = torch.optim.Adam(task.parameters(), lr=1e-6)
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.1, 
            patience=35
        )
        early_stopping = EarlyStopping(patience=65, verbose=True)
        
        # Path setup
        best_model_path = os.path.join(config.output_dir, f'nodupes_trim_best_full_3_{fold_idx}.pth')
        final_model_path = os.path.join(config.output_dir, f'nodupes_trim_last_full_3_{fold_idx}.pth')
        
        # Initialize engine
        solver = core.Engine(
            task=task,
            train_set=train_set,
            valid_set=valid_set,
            test_set=None,
            optimizer=optimizer,
            scheduler=scheduler,
            early_stopping=early_stopping,
            best_model_path=best_model_path,
            batch_size=35,
            gpus=[0]
        )
        
        # Training
        solver.train(num_epoch=config.num_epochs)
        
        # Save final model with verification
        solver.save(final_model_path)
        if not os.path.exists(final_model_path):
            raise RuntimeError(f"Failed to save model at {final_model_path}")
        print(f"Successfully saved final model to {final_model_path}")
        
        # Robust evaluation with fallbacks
        try:
            metrics = solver.evaluate("valid")
            
            # Debug: Print all available metrics
            print("\nAvailable metrics:", metrics)
            
            # Get accuracy with multiple possible keys
            accuracy = metrics.get('acc', metrics.get('accuracy', float('nan')))
            
            # Get loss if available
            val_loss = metrics.get('loss', float('nan'))
            
            print(f"\nFold {fold_idx} Validation Metrics:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Loss: {val_loss:.4f}")
            
            # Log all metrics plus paths
            log_data = {
                "final_val_accuracy": accuracy,
                "final_val_loss": val_loss,
                "best_model_path": best_model_path,
                "final_model_path": final_model_path,
                **metrics  # Include all original metrics
            }
            wandb.log(log_data)
            
        except Exception as e:
            print(f"\n⚠️ Evaluation failed! Error: {str(e)}")
            wandb.log({
                "final_val_accuracy": float('nan'),
                "final_val_loss": float('nan'),
                "error": str(e),
                "best_model_path": best_model_path,
                "final_model_path": final_model_path
            })
        
        # Cleanup
        wandb.finish()
        del solver
        torch.cuda.empty_cache()