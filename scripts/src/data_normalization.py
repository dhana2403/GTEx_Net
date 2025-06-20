import os
import pandas as pd
import pickle
import numpy as np

def calcNormFactors(readcounts):
    library_size = readcounts.sum(axis=0)
    norm_factor = library_size / np.median(library_size)
    return readcounts.div(norm_factor, axis=1)

def calcCPM(readcounts):
    library_size = readcounts.sum(axis=0)
    cpm = readcounts.div(library_size, axis=1) * 1e6  # proper CPM scale
    return cpm

def normalize_readcounts_for_tissue(tis, attphe_path, input_dir, output_dir, sample_counts_list):
    attphe = pd.read_pickle(attphe_path)

    readcounts_path = os.path.join(input_dir, f"{tis}.pkl")
    readcounts = pd.read_pickle(readcounts_path)

    intersected_samples = set(readcounts.columns).intersection(attphe['samp_id'])

    if not intersected_samples:
        print(f"Skipping {tis}: no overlapping samples.")
        return

    readcounts_filtered = readcounts[list(intersected_samples)]

    readcounts_tmm = calcNormFactors(readcounts_filtered)
    readcounts_cpm = calcCPM(readcounts_tmm)
    readcounts_log2cpm = np.log2(readcounts_cpm + 1)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, tis)
    readcounts_log2cpm.to_pickle(output_path)

    sample_counts_list.append({
        'Tissue': tis,
        'SampleCount': len(intersected_samples)
    })

    print(f'Normalized data saved for {tis}')