import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import os

# Path to metadata
meta_path = "/Users/dhanalakshmijothi/Desktop/python/GTEx_Net/scripts/data/processed/attphe.pkl"
meta_df = pd.read_pickle(meta_path)

# Combine batch columns to one label (optional)
meta_df['batch'] = meta_df['batch_iso'].astype(str) + '_' + meta_df['batch_exp'].astype(str)

# Path where batch corrected expression files (.pkl) are stored, one per tissue
expr_folder = "/Users/dhanalakshmijothi/Desktop/python/GTEx_Net/scripts/data/processed/batch_corrected/"

results = []

# Get unique tissues from metadata
tissues = meta_df['minor_tissue'].unique()

for tissue in tissues:
    try:
        # Load expression data for this tissue
        expr_path = os.path.join(expr_folder, f"{tissue}_corrected.pkl")
        expr_df = pd.read_pickle(expr_path)
        
        meta_samp_ids = set(meta_df['samp_id'])
        samples_in_both = [s for s in expr_df.index if s in meta_samp_ids]
        print(f"Matched samples: {len(samples_in_both)}")
        tissue_meta = meta_df.set_index('samp_id').loc[samples_in_both]
        
        # Add batch info to expression DataFrame
        expr_df['batch'] = tissue_meta['batch']
        
        # Calculate average expression per batch for this tissue
        avg_expr = expr_df.groupby('batch').mean()
        
        batches = avg_expr.index.tolist()
        
        # Pearson correlation between batch averages
        for i, batch1 in enumerate(batches):
            for batch2 in batches[i+1:]:
                vec1 = avg_expr.loc[batch1].values
                vec2 = avg_expr.loc[batch2].values
                corr, _ = pearsonr(vec1, vec2)
                results.append({
                    'tissue': tissue,
                    'batch1': batch1,
                    'batch2': batch2,
                    'pearson_corr': corr
                })
    
    except FileNotFoundError:
        print(f"Expression file for tissue {tissue} not found, skipping.")

corr_df = pd.DataFrame(results)
summary = corr_df.groupby('tissue')['pearson_corr'].describe().reset_index()
summary.to_csv("/Users/dhanalakshmijothi/Desktop/python/GTEx_Net/summary.csv", index=False)


