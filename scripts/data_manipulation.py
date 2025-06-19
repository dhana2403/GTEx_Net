
######################### DATA MANIPULATION ###############################

##############################ATTRIBUTES####################################

import pandas as pd 


att_des = (
           pd.read_excel('/Users/dhanalakshmijothi/Desktop/python/GTEx/data/metadata/GTEx_Analysis_v8_Annotations_SampleAttributesDD.xlsx')
           .fillna("")
)

att_val = (
           pd.read_csv('/Users/dhanalakshmijothi/Desktop/python/GTEx/data/metadata/GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt',  sep='\t')
           .fillna("")
           .rename(columns={'SAMPID':'samp_id','SMRIN':'rin_val', 'SMTS': 'major_tissue', 'SMTSD': 'minor_tissue', 'SMNABTCH':'batch_iso', 'SMGEBTCH':'batch_exp', 'SMTSISCH': 'ischemic_time'})
           .assign(minor_tissue = lambda df: df['minor_tissue'].str.replace(' - ', '-', regex=False),
                   subj_id = lambda df: df['samp_id'].str.split('-', n=2).str[:2].apply('-'.join))
)

att_val_new = att_val[['samp_id','subj_id','rin_val','major_tissue','minor_tissue','batch_iso','batch_exp', 'ischemic_time']]

att_des, att_val, att_val_new
##############################PHENOTYPES####################################

phe_des = (
    pd.read_excel('/Users/dhanalakshmijothi/Desktop/python/GTEx/data/metadata/GTEx_Analysis_v8_Annotations_SubjectPhenotypesDD.xlsx')
    .fillna("")
)

phe_val = (
    pd.read_csv('/Users/dhanalakshmijothi/Desktop/python/GTEx/data/metadata/GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt',  sep='\t')
    .fillna("")  
    .rename(columns={'SUBJID': 'subj_id', 'SEX': 'sex', 'AGE': 'age', 'DTHHRDY': 'Death_category'})
    .assign(sex = lambda df: df['sex'].replace({1: "male", 2: "female"}),
            Death_category = lambda df: df['Death_category'].replace({0: "Ventilator", 1: "Fast & violent", 2: "Fast & natural", 3: "Intermediate", 4: "slow death"}))
)
##########################JOIN ATTRIBUTES AND PHENOTYPES##########################

attphe = (att_val_new
          .merge(phe_val, on='subj_id', how='inner')
          .drop_duplicates(subset='samp_id', keep='first')
           # .loc[lambda df: df['minor_tissue'].isin(['Brain-Cortex', 'Lung', 'Liver'])]
          )
###################RAW DATA################

import pandas as pd
import re
import os
import pickle

# Load the gene expression data (adjust file path accordingly)
raw_dat = pd.read_csv('/Users/dhanalakshmijothi/Desktop/python/GTEx/data/raw/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_reads.gct', sep='\t', skiprows=2, header=0)

# Extract sample IDs from metadata (attphe should be defined beforehand)
samp_ids = attphe['samp_id'].to_list()

# Define columns to keep (including 'Name' and columns that match sample IDs)
columns_to_keep = ['Name'] + [col for col in raw_dat.columns if col in samp_ids]

# Reload raw data with the selected columns
raw_dat = pd.read_csv('/Users/dhanalakshmijothi/Desktop/python/GTEx/data/raw/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_reads.gct', sep='\t', skiprows=2, header=0, usecols=columns_to_keep)

# Define genes of interest
genes_of_interest = ['ENSG00000198793', 'ENSG00000118689', 'ENSG00000096717', 'ENSG00000142082', 'ENSG00000133818', 'ENSG00000121691', 'ENSG00000017427', 
                     'ENSG00000140443', 'ENSG00000141510', 'ENSG00000077463', 'ENSG00000130203', 'ENSG00000126458', 'ENSG00000142168', 'ENSG00000133116']

# Clean gene names (remove version numbers) and filter for genes of interest
raw_dat = raw_dat.fillna("").dropna()
raw_dat['Name'] = raw_dat['Name'].str.replace(r"\.\d+$", "", regex=True)
raw_dat = raw_dat[raw_dat['Name'].isin(genes_of_interest)]

# Group samples by tissue (using metadata `attphe`)
sample_meta = attphe.groupby("minor_tissue")["samp_id"].apply(list).to_dict()

# Filter the raw data columns based on tissue samples
sample_raw = list(raw_dat.columns)
sample_meta = {tissue: list(set(samps) & set(sample_raw)) for tissue, samps in sample_meta.items()}

# Filter the data for each tissue group (ensure each tissue group has at least 2 samples)
dat_filtered_tissues = {
    tissue: raw_dat[sample].dropna(axis=1) 
    for tissue, sample in sample_meta.items() 
    if len(sample) > 1
}

# Clean the row names 
dat_filtered_tissues = {
    tissue: df.rename(columns=lambda x: re.split(r'[()]', x.replace(' ', ''))[0]) 
    for tissue, df in dat_filtered_tissues.items()
}

# Create directory if it does not exist
save_dir = "data/processed/expression/readcounts_all/"
save_dir1 = "data/processed/"
os.makedirs(save_dir, exist_ok=True)
os.makedirs(save_dir1, exist_ok=True)

# Save the filtered data (columns by tissue)
for tissue, df in dat_filtered_tissues.items():
    pickle.dump(df, open(os.path.join(save_dir, f"{tissue}.pkl"), "wb"))

# Save metadata (attphe)
pickle.dump(attphe, open(os.path.join(save_dir1, "attphe.pkl"), "wb"))

print(f"Files saved in: {save_dir} and {save_dir1}")
