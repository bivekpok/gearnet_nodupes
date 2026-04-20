import os
import torch
from torch.utils import data as torch_data
from torchdrug import data, layers, models, tasks, transforms, utils
from torchdrug.data.dataloader import graph_collate
from torchdrug.layers import geometry
from torchdrug import core
# ==========================================
# 1. EXACT SAME MODEL INIT FROM YOUR CODE
# ==========================================
def initialize_model(
    num_classes: 9, 
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
# ==========================================
# 2. STRIPPED-DOWN INFERENCE DATALOADER
# ==========================================
class SingleProteinDataset(torch_data.Dataset):
    """A minimal dataset just for one PDB file so TorchDrug can build the MLP."""
    def __init__(self, graph):
        self.graph = graph
        # Apply the EXACT same transform used in training
        self.transform = transforms.ProteinView(view='residue')

    def __len__(self):
        return 1

    def __getitem__(self, index):
        # We pass a dummy label of 0 because we are only predicting, not training
        item = {'graph': self.graph, 'label': 0}
        return self.transform(item)

def load_pdb_graph(pdb_path: str):
    """Applies the exact same AlphaCarbon filtering as your training _load_proteins."""
    iprotein = data.Protein.from_pdb(
        pdb_path, 
        atom_feature="position", 
        bond_feature="length", 
        residue_feature="symbol"
    )
    
    filter_alpha = layers.GraphConstruction(node_layers=[geometry.AlphaCarbonNode()])
    protein = filter_alpha(iprotein)
    
    if protein is None:
        raise ValueError(f"Failed to load alpha-carbon graph from {pdb_path}")
    return protein

# ==========================================
# 3. INFERENCE FUNCTION
# ==========================================
# ==========================================
# 3. INFERENCE FUNCTION
# ==========================================
def run_local_inference(pdb_path: str, checkpoint_path: str, config: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on device: {device}")

    # 1. Load the PDB file and create the single-item dataset
    print(f"Loading PDB: {pdb_path}...")
    graph = load_pdb_graph(pdb_path)
    dataset = SingleProteinDataset(graph)

    # 2. Initialize the model architecture
    print("Initializing model architecture...")
    task = initialize_model(
        num_classes=config["num_classes"],
        mlp_dropout=config.get("mlp_dropout", 0.2),
        hidden_dim=config.get("hidden_dim", 512),
        num_gearnet_layers=config.get("num_gearnet_layers", 3),
        readout=config.get("readout", "sum"),
        concat_hidden=config.get("concat_hidden", False),
        knn_k=config.get("knn_k", 10),
        spatial_radius=config.get("spatial_radius", 10.0),
        activation=config.get("activation", "relu")
    )

    # 3. Create a Dummy Engine to utilize TorchDrug's native loader
    print("Initializing TorchDrug Engine...")
    solver = core.Engine(
        task=task,
        train_set=dataset, # Passed here so Engine natively handles MLP building
        valid_set=dataset, 
        test_set=dataset,
        optimizer=None,    # No optimizer needed for inference
        early_stopping=None,
        best_model_path=None, 
        batch_size=1,
        gpus=[0] if device.type == "cuda" else None
    )

    # 4. Load the weights EXACTLY how you do in training
    print(f"Loading weights via solver from {checkpoint_path}...")
    solver.load(checkpoint_path)

    # 5. Predict
    print("Running final prediction...")
    task.eval() # Ensure dropout and batchnorm are in eval mode
    loader = torch_data.DataLoader(dataset, batch_size=1, collate_fn=graph_collate)
    batch = next(iter(loader))
    
    if device.type == "cuda":
        batch = utils.cuda(batch, device=device)

    with torch.no_grad():
        pred = task.predict(batch)
        probs = torch.softmax(pred, dim=-1)

    print("\n=== PREDICTION RESULTS ===")
    print(f"Raw Logits: {pred.cpu().numpy()}")
    print(f"Probabilities: {probs.cpu().numpy()}")
    
    predicted_class = torch.argmax(probs, dim=-1).item()
    print(f"=> Model predicts Class Index: {predicted_class}")
# ==========================================
# 4. RUN THE TEST
# ==========================================
if __name__ == "__main__":
    # --- UPDATE THESE PATHS BEFORE RUNNING ---
    TEST_PDB = "/work/hdd/bdja/bpokhrel/pdb_ac_missing/1a0s"
    CHECKPOINT = "/work/hdd/bdja/bpokhrel/new_gearnet/foldseek_train/whole_test_cover/production_models_v2/Outer_Fold_1_Inner_Fold_1_best_model.pth"
    
    # Update this dictionary to match EXACTLY what you trained the model with
    MY_TRAINING_CONFIG = {
        "num_classes": 9,  # <--- Make sure this matches your actual number of classes!
        "hidden_dim": 512,
        "num_gearnet_layers": 5,
        "mlp_dropout": 0.2903210512935248,
        "readout": "mean",
        "concat_hidden": False, 
        "knn_k": 25,
        "spatial_radius": 12.0,
        "activation": "relu"
    }
    if os.path.exists(TEST_PDB) and os.path.exists(CHECKPOINT):
        run_local_inference(TEST_PDB, CHECKPOINT, MY_TRAINING_CONFIG)
    else:
        print("Please update the TEST_PDB and CHECKPOINT variables with real file paths!")