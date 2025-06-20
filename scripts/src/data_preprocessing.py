import pandas as pd  # type: ignore
import os
import pickle
import re

def preprocess_metadata(file_path):
     """
    Load attribute and phenotype files and clean them
    """

     if "Attributes" in file_path:
         att_val = (
           pd.read_csv(file_path, sep='\t')
           .fillna("")
           .rename(columns={'SAMPID':'samp_id','SMRIN':'rin_val', 'SMTS': 'major_tissue', 'SMTSD': 'minor_tissue', 'SMNABTCH':'batch_iso', 'SMGEBTCH':'batch_exp', 'SMTSISCH': 'ischemic_time'})
           .assign(minor_tissue = lambda df: df['minor_tissue'].str.replace(' - ', '-', regex=False),
                   subj_id = lambda df: df['samp_id'].str.split('-', n=2).str[:2].apply('-'.join)))
         return att_val[['samp_id', 'subj_id', 'rin_val', 'major_tissue', 'minor_tissue', 'batch_iso', 'batch_exp', 'ischemic_time']]

     elif "Phenotypes" in file_path:
          
         phe_val = (
            pd.read_csv(file_path, sep='\t')
            .fillna("")
            .rename(columns={'SUBJID': 'subj_id', 'SEX': 'sex', 'AGE': 'age', 'DTHHRDY': 'Death_category'})
            .assign(sex = lambda df : df['sex'].replace({1:"male", 2:"female"}), 
                    Death_category = lambda df: df['Death_category'].replace({0: "Ventilator", 1: "Fast & violent", 2: "Fast & natural", 3: "Intermediate", 4: "slow death"}))
                )
         return phe_val[['subj_id', 'sex', 'age', 'Death_category']]
    
     else:
         raise ValueError("File path must contain 'Attributes' or 'Phenotypes' to be recognized.")
    

def load_and_filter_expression (rawfile_path, sample_ids, genes_of_interest):
    
      """
     Load raw gene expression data, filter by sample IDs and genes of interest.
     """ 
      print("Reading header to identify columns...")
      
      # Read only the header line to get all column names
      cols = pd.read_csv(rawfile_path, sep='\t', skiprows=2, nrows=0).columns.tolist()

      # Columns to keep: "Name" + sample_ids that exist in the file
      columns_to_keep = ['Name'] + [col for col in sample_ids if col in cols]
      print(f"Columns to keep: {len(columns_to_keep)} total")

       # Read only those columns and save to new file
      print("Reading and saving filtered columns...")
      raw_dat = pd.read_csv(rawfile_path, sep='\t', skiprows=2, usecols=columns_to_keep)
      print(f"Loaded data shape: {raw_dat.shape}")
      raw_dat['Name'] = raw_dat['Name'].str.replace(r"(\.\d+)$", "", regex=True)
      raw_dat = raw_dat[raw_dat['Name'].isin(genes_of_interest)].set_index('Name')

      return raw_dat

def group_sample_by_tissues(metadata_df, raw_dat):
         """
         Group samples by tissue and filter raw expression data accordingly.
         Returns a dict of {tissue: DataFrame of expression data}.
         """
         sample_meta = metadata_df.groupby("minor_tissue")["samp_id"].apply(list).to_dict()
         sample_raw = set(raw_dat.columns)

         # Filter samples in raw data for each tissue
         sample_meta = {tissue: list(set(samples) & sample_raw) for tissue, samples in sample_meta.items()}

         # Keep tissues with at least 2 samples
         dat_filtered_tissues = {
           tissue: raw_dat[samples].dropna(axis=1)
           for tissue, samples in sample_meta.items()
           if len(samples)>1
            }

         dat_filtered_tissues = { 
          tissue: df.rename(columns= lambda x: re.split(r'[()]', x.replace('', ''))[0])
          for tissue, df in dat_filtered_tissues.items()  
          }

         return dat_filtered_tissues

def save_dataframes(dataframes_dict, save_dir, metadata_df=None, metadata_path=None):
         """
         Save expression data dict as pickles; optionally save metadata.
         """
         os.makedirs(save_dir, exist_ok=True)

         for tissue, df in dataframes_dict.items():
           pickle.dump(df, open(os.path.join(save_dir, f"{tissue}.pkl"), "wb"))

         if metadata_df is not None and metadata_path is not None:
            with open (metadata_path, "wb") as f:
             pickle.dump(metadata_df, f)
      
def main(att_path, phe_path, raw_path):
      print("hi, im inside main of data_preprocessing")

      """
      preprocess attributes, phenotype and raw data files
      """
      att_val_new = preprocess_metadata(att_path)
      print("Attributes loaded.")
      phe_val = preprocess_metadata(phe_path)
      print("phenotypes loaded.")

      attphe = (
       att_val_new
      .merge(phe_val, on='subj_id', how='inner')
      .drop_duplicates(subset='samp_id', keep='first')
      )

      samp_ids = attphe['samp_id'].to_list()

      genes_of_interest = [
      'ENSG00000198793', 'ENSG00000118689', 'ENSG00000096717', 'ENSG00000142082',
      'ENSG00000133818', 'ENSG00000121691', 'ENSG00000017427', 'ENSG00000140443',
      'ENSG00000141510', 'ENSG00000077463', 'ENSG00000130203', 'ENSG00000126458',
      'ENSG00000142168', 'ENSG00000133116']
    
      raw_dat = load_and_filter_expression(raw_path, samp_ids, genes_of_interest)
      print("Raw data filtered")

      dat_filtered_tissues = group_sample_by_tissues(attphe, raw_dat)

      save_dir = "data/processed/expression/readcounts_all/"
      metadata_path = "data/processed/attphe.pkl"
      save_dataframes(dat_filtered_tissues, save_dir, metadata_df = attphe, metadata_path= metadata_path)

      print(f"Processed data saved to:\n- {save_dir}\n- {metadata_path}")