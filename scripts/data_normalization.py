
######################### DATA NORMALIZATION ###############################

import os
import pandas as pd
import pickle
import numpy as np

def calcNormFactors(readcounts):

    library_size = readcounts.sum(axis = 0)
    norm_factor = library_size/np.median(library_size)
    return readcounts.div(norm_factor, axis = 1)

def calcCPM(readcounts):

    library_size = readcounts.sum(axis=0)
    cpm = readcounts.div(library_size, axis=1) * 1e6  # CPM normalization
    return cpm

def readcounts_norm(tis):

    attphe = pd.read_pickle('data/processed/attphe.pkl')
    
    readcounts = pd.read_pickle(f'./data/processed/expression/readcounts_all/{tis}')

    intersected_samples = set(readcounts.columns).intersection(attphe['samp_id'])

    attphe_filtered = attphe[attphe['samp_id'].isin(intersected_samples)]

    readcounts_filtered = readcounts[list(intersected_samples)]

    readcounts_tmm = calcNormFactors(readcounts_filtered)

    readcounts_cpm = calcCPM(readcounts_filtered)
    
    readcounts_log2cpm = np.log2(readcounts_cpm + 1)


    output_dir = "./data/processed/expression/readcounts_tmm_all/"
    
    os.makedirs(output_dir, exist_ok=True)

    readcounts_log2cpm.to_pickle(os.path.join(output_dir, tis)) 

    sample_counts_list.append ({
        'Tissue' : tis,
        'SampleCount': len(intersected_samples)

    })

    print(f'Normalized data saved for {tis}')

alltis = os.listdir('data/processed/expression/readcounts_all/')

sample_counts_list = []

for tis in alltis:
    readcounts_norm(tis)


global_sample_counts_df = pd.DataFrame(sample_counts_list)
global_sample_counts_df.to_csv('./data/processed/sample_counts.csv', index=False)
