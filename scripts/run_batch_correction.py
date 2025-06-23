import os
import pickle
import numpy as np
import pandas as pd
from src.auto1 import train_autoencoder_supervised, get_latent_representation

def run_batch_correction(input_dir, output_dir, metadata_path, device='cpu'):
    os.makedirs(output_dir, exist_ok=True)

    metadata = pd.read_pickle(metadata_path)
    metadata = metadata.set_index('samp_id')

    tissue_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.pkl')])

    all_samples_dfs = []
    all_tissue_labels = []
    all_batch_ids = []

    # Load and combine all tissues
    for filename in tissue_files:
        tissue_name = filename.replace('.pkl', '')
        filepath = os.path.join(input_dir, filename)
        print(f"Loading tissue: {tissue_name}")

        df = pickle.load(open(filepath, 'rb')).T
        samples = df.index.tolist()

        # Filter metadata for samples in this tissue
        meta_subset = metadata.loc[metadata.index.intersection(samples)]

        if meta_subset.empty:
            print(f"No metadata matches for tissue {tissue_name}, skipping.")
            continue

        df = df.loc[meta_subset.index]
        all_samples_dfs.append(df)

        # Collect tissue and batch labels aligned to samples
        all_tissue_labels.extend([tissue_name] * len(df))
        all_batch_ids.extend(meta_subset[['batch_iso', 'batch_exp']].astype(str).agg('_'.join, axis=1).values)

    # Combine all tissues into one DataFrame
    combined_df = pd.concat(all_samples_dfs)
    X = combined_df.values
    tissue_labels = np.array(all_tissue_labels)
    batch_ids = np.array(all_batch_ids)

    print("Training autoencoder on combined data...")
    model, scaler, tissue_encoder, batch_encoder = train_autoencoder_supervised(
        X, tissue_labels, batch_ids, device=device
    )

    print("Generating latent representations...")
    latent_all = get_latent_representation(
        model, X, scaler, device=device
    )

    # Split latent embeddings back by tissue and save
    combined_df['tissue'] = tissue_labels
    combined_df['batch'] = batch_ids
    combined_df['index'] = combined_df.index

    latent_df = pd.DataFrame(latent_all, index=combined_df.index)

    for tissue in np.unique(tissue_labels):
        tissue_idx = combined_df['tissue'] == tissue
        latent_subset = latent_df.loc[tissue_idx.index[tissue_idx]]
        
        save_path = os.path.join(output_dir, f"{tissue}_corrected.pkl")
        with open(save_path, 'wb') as f:
            pickle.dump(latent_subset, f)
        print(f"Saved corrected latent data for tissue {tissue} at {save_path}")

if __name__ == "__main__":
    input_dir = "/Users/dhanalakshmijothi/Desktop/python/GTEx_Net/scripts/data/processed/expression/readcounts_all"
    output_dir = "/Users/dhanalakshmijothi/Desktop/python/GTEx_Net/scripts/data/processed/batch_corrected"
    metadata_path = "/Users/dhanalakshmijothi/Desktop/python/GTEx_Net/scripts/data/processed/attphe.pkl"

    run_batch_correction(input_dir, output_dir, metadata_path, device='cpu')
