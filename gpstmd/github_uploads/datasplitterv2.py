import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, train_test_split

def print_split_stats(df, split_name):
    """Helper function to print the population and class ratios of a dataframe."""
    total = len(df)
    # Sort by index to ensure Class 0 and Class 1 are always printed in the same order
    counts = df['label'].value_counts().sort_index()
    ratios = df['label'].value_counts(normalize=True).sort_index() * 100
    
    # Format a neat, aligned string
    stats_str = f"{split_name:<30} | Total: {total:<5} | "
    class_stats = []
    for cls in counts.index:
        class_stats.append(f"Class {cls}: {counts[cls]} ({ratios[cls]:.1f}%)")
    
    stats_str += " | ".join(class_stats)
    print(stats_str)

def generate_hybrid_splits(csv_path, output_root, n_outer=6, n_inner_for_tune=3):
    df = pd.read_csv(csv_path)
    os.makedirs(output_root, exist_ok=True)

    print("="*85)
    print_split_stats(df, "FULL ORIGINAL DATASET")
    print("="*85)

    # 1. OUTER LOOP: Create the 6 Test Sets
    outer_skf = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=65)

    for out_idx, (remainder_idx, test_idx) in enumerate(outer_skf.split(df, df['label'])):
        outer_num = out_idx + 1
        outer_path = os.path.join(output_root, f"Outer_Fold_{outer_num}")
        os.makedirs(outer_path, exist_ok=True)
        
        # Save the Test Set
        test_df = df.iloc[test_idx]
        test_df.to_csv(os.path.join(outer_path, "test_manifest.csv"), index=False)
        remainder_df = df.iloc[remainder_idx].reset_index(drop=True)

        print(f"\n--- OUTER FOLD {outer_num} ---")
        print_split_stats(test_df, f"Outer {outer_num} TEST SET")

        # 2. INNER LOOP LOGIC
        if outer_num == 1:
            # --- FOLD 1: Create 3 Inner Folds for Hyperparameter Sweeping ---
            # Swapped to StratifiedShuffleSplit so we can force a 10% val size while keeping 3 splits
            inner_sss = StratifiedShuffleSplit(n_splits=n_inner_for_tune, test_size=0.08, random_state=65)
            
            for in_idx, (train_idx, val_idx) in enumerate(inner_sss.split(remainder_df, remainder_df['label'])):
                inner_path = os.path.join(outer_path, f"Inner_Fold_{in_idx + 1}")
                os.makedirs(inner_path, exist_ok=True)
                
                train_df = remainder_df.iloc[train_idx]
                val_df = remainder_df.iloc[val_idx]
                
                train_df.to_csv(os.path.join(inner_path, "train_manifest.csv"), index=False)
                val_df.to_csv(os.path.join(inner_path, "valid_manifest.csv"), index=False)
                
                print_split_stats(train_df, f"  Inner {in_idx + 1} TRAIN SET")
                print_split_stats(val_df, f"  Inner {in_idx + 1} VALID SET")
            
        else:
            # --- FOLDS 2 to 6: Create exactly 1 Inner Fold for Final Training ---
            # Shrunk test_size from 0.20 down to 0.10
            train_df, val_df = train_test_split(
                remainder_df, 
                test_size=0.08, 
                random_state=65, 
                stratify=remainder_df['label']
            )
            
            inner_path = os.path.join(outer_path, "Inner_Fold_1")
            os.makedirs(inner_path, exist_ok=True)
            
            train_df.to_csv(os.path.join(inner_path, "train_manifest.csv"), index=False)
            val_df.to_csv(os.path.join(inner_path, "valid_manifest.csv"), index=False)
            
            print_split_stats(train_df, f"  Inner 1 TRAIN SET")
            print_split_stats(val_df, f"  Inner 1 VALID SET")

if __name__ == "__main__":
    
    generate_hybrid_splits(
        "/work/hdd/bdja/bpokhrel/new_gearnet/foldseek_train/fs_sl_df_noGolgiLysosomeVacuole.csv", 
        "/work/hdd/bdja/bpokhrel/new_gearnet/foldseek_train/whole_test_cover/production_splitsv2"
    )