import os
import pickle
import torch
from src.autoencoder import Autoencoder, scale_data, train_autoencoder, get_latent_representation
import pandas as pd

def run_batch_correction(input_dir, output_dir, device='cpu'):
    os.makedirs(output_dir, exist_ok=True)

    for filename in sorted(os.listdir(input_dir)):
        if not filename.endswith('.pkl'):
            continue

        tissue_name = filename.replace('.pkl', '')
        filepath = os.path.join(input_dir, filename)

        print(f"Processing tissue: {tissue_name}")

        # Load and transpose the dataframe (genes x samples → samples x genes)
        df = pickle.load(open(filepath, 'rb')).T
        X = df.values  # numpy array

        # Train autoencoder and get batch-corrected latent space
        model, scaler = train_autoencoder(X, device=device)
        X_corrected = get_latent_representation(model, X, scaler, device=device)

        # Convert corrected numpy array back to DataFrame
        corrected_df = pd.DataFrame(X_corrected, index=df.index)
        
        # Save as .pkl
        save_path = os.path.join(output_dir, f"{tissue_name}_corrected.pkl")
        with open(save_path, 'wb') as f:
            pickle.dump(corrected_df, f)

        print(f"Saved corrected data for {tissue_name} at {save_path}")

if __name__ == "__main__":
        input_dir = "/Users/dhanalakshmijothi/Desktop/python/GTEx_Net/scripts/data/processed/expression/readcounts_all"
        output_dir= "/Users/dhanalakshmijothi/Desktop/python/GTEx_Net/scripts/data/processed/batch_corrected"
        run_batch_correction(input_dir, output_dir, device='cpu')
