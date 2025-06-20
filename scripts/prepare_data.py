import os
from src import data_preprocessing  # type: ignore
from src import data_normalization  # type: ignore

def prepare_main(att_path, phe_path, raw_path, processed_dir):
    
      print("hi im inside main of prepare data")
    
      # Step 1: Preprocess raw data (metadata + expression)
      data_preprocessing.main(att_path, phe_path, raw_path)
      print("Returned from data_preprocessing.main()")

      # Step 2: Normalize data per tissue
      attphe_path = os.path.join('data/processed', 'attphe.pkl')
      input_dir = os.path.join('data', 'processed', 'expression', 'readcounts_all')
      output_dir = os.path.join(processed_dir, 'normalized')

      # We assume all tissues are saved as pickle files in input_dir
      tissue_files = [f for f in os.listdir(input_dir) if f.endswith('.pkl')]
 
      sample_counts_list = []

      for tissue_file in tissue_files:
        tissue = tissue_file.replace('.pkl', '')
        data_normalization.normalize_readcounts_for_tissue(tissue, attphe_path, input_dir, output_dir, sample_counts_list)

      print("Data preparation complete.")

if __name__ == "__main__":
    
      att_path = '/Users/dhanalakshmijothi/Desktop/python/GTEx_Net/data/metadata/GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt'
      phe_path = '/Users/dhanalakshmijothi/Desktop/python/GTEx_Net/data/metadata/GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt'
      raw_path = '/Users/dhanalakshmijothi/Desktop/python/GTEx_Net/data/raw/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_reads.gct'
      processed_dir = '/Users/dhanalakshmijothi/Desktop/python/GTEx_Net/data/processed'
    
      prepare_main(att_path, phe_path, raw_path, processed_dir)
