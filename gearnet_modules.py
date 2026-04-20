# gearnet_modules.py
import os
import random
import torch
import numpy as np
from torch.utils import data as torch_data

from torchdrug import models, transforms, data, layers, tasks
from torchdrug.layers import geometry

# Heavy deps (pandas + sklearn) are only needed by ProteinLoader during
# training/evaluation. Inference-only callers (e.g. the Streamlit app on
# Hugging Face Spaces) can import `initialize_model` and `transform` without
# paying for them.

# Define the transform globally for the dataset
transform = transforms.ProteinView(view='residue')

class ProteinLoader(data.ProteinDataset):
    """Custom dataset loader for protein data with membrane classification."""
    
    def __init__(self, csv_path: str, pdb_folder: str, soluble_folder_ac: str, transform, random_seed: int = None, balance: bool = False):
        self.pdb_folder = pdb_folder
        self.soluble_folder_ac = soluble_folder_ac
        self.balance = balance
        self.random_seed = random_seed
        
        if random_seed is not None:
            np.random.seed(random_seed)
            
        self.csv_path = csv_path
        self.transform = transform
        self.label_dict, self.df, self.k, self.label_population, self.wt = self._make_dict()
        self.pdb_list = self.df['pdbid'].tolist()
        self.pdbs = self.df['pdbid']
        
        self.data, self.pdb_list2, self.pdb_list3, self.pdb_name = self._load_proteins()
        self.label_list = self._get_labels()
        # self.label_list = [random.choice(range(self.k)) for _ in range(len(self.label_list))]  ### for random label ###
        self.targets = {'label': self.label_list}
        self.num_samples = len(self.data)
        
    def _make_dict(self):
        import pandas as pd  # lazy: only needed for training/eval

        try:
            df = pd.read_csv(self.csv_path)
            counts = df['label'].value_counts()
            label_dict_counts = {key: counts[key] for key in counts.index}
            print('Label distribution:', label_dict_counts)
            
            location = df['label'].unique()
            label_dict = {key: value for value, key in enumerate(sorted(location))}
            
            if self.balance:
                def select_100_rows(group):
                    return group.sample(n=100, random_state=self.random_seed) if len(group) >= 100 else group
                df = df.groupby('label', group_keys=False).apply(select_100_rows).reset_index(drop=True)
                
            wt = torch.zeros(len(label_dict))
            for label, idx in label_dict.items():
                wt[idx] = 1.0 / counts[label]
            wt = wt / wt.mean()
            
            # --- QUICK HACK: EQUAL WEIGHTS FOR RANDOM LABELS---
            # wt = torch.ones(len(label_dict))
            # --------------------------------
            
            return label_dict, df, len(label_dict), label_dict_counts, wt
            
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            raise
            
    def _load_proteins(self):
        protein_list = []
        pdb_name = []
        pdb_list2 = [] 
        pdb_list3 = [] 
        
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
        return [
            self.label_dict[self.df.loc[self.df['pdbid'] == pdb, 'label'].values[0]]
            for pdb in self.pdbs 
            if pdb not in self.pdb_list2 and pdb not in self.pdb_list3
        ]
        
    def create_stratified_splits(self, test_size=0.15, n_splits=3):
        """Creates a stratified test set and stratified K-Folds from the remainder."""
        from sklearn.model_selection import StratifiedKFold, train_test_split  # lazy

        all_indices = list(range(len(self.data)))
        all_labels = self.targets["label"]
        
        # 1. Stratified Train/Test Split
        train_val_idx, test_idx = train_test_split(
            all_indices, 
            test_size=test_size, 
            stratify=all_labels, 
            random_state=self.random_seed
        )
        
        test_set = torch_data.Subset(self, test_idx)
        print(f"Reserved {len(test_set)} samples for the final Test Set.")
        
        # 2. Stratified K-Fold on the remaining train_val data
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_seed)
        train_val_labels = [all_labels[i] for i in train_val_idx]
        
        folds = []
        for relative_train_idx, relative_val_idx in skf.split(train_val_idx, train_val_labels):
            abs_train_idx = [train_val_idx[i] for i in relative_train_idx]
            abs_val_idx = [train_val_idx[i] for i in relative_val_idx]
            
            folds.append((
                torch_data.Subset(self, abs_train_idx),
                torch_data.Subset(self, abs_val_idx)
            ))
            
        print(f"Created {n_splits} folds from the remaining {len(train_val_idx)} samples.")
        return folds, test_set
        
    def __getitem__(self, index):
        protein = self.data[index]
        label = self.label_list[index]
        name = self.pdb_name[index]
        item = {'graph': protein, 'label': label, 'name': name}
        return self.transform(item) if self.transform else item
        
    def __len__(self):
        return len(self.data)

class EarlyStopping:
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

def initialize_model(
    num_classes: int, 
    class_weights=None, 
    mlp_dropout=0.2, 
    hidden_dim=512,
    num_gearnet_layers=3,     
    readout="sum",            
    concat_hidden=False,      
    knn_k=10,                 
    spatial_radius=10.0,
    activation="relu"   
):
    """Initialize the GearNet model with dynamic sweep parameters."""
    
    hidden_dims_list = [hidden_dim] * num_gearnet_layers
    
    gearnet_edge = models.GearNet(
        input_dim=21, 
        hidden_dims=hidden_dims_list,
        num_relation=7,
        batch_norm=True, 
        concat_hidden=concat_hidden,  
        short_cut=True, 
        readout=readout,
        activation=activation             
    )
    
    graph_construction_model = layers.GraphConstruction(
        node_layers=[geometry.AlphaCarbonNode()], 
        edge_layers=[
            geometry.SpatialEdge(radius=spatial_radius, min_distance=5),
            geometry.KNNEdge(k=knn_k, min_distance=5),                   
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
        mlp_dropout=mlp_dropout,
        task=['label'], 
        criterion="ce", 
        metric=["acc"],
        class_weights=class_weights
    )
    
    return task