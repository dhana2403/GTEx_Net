import os
import requests
import subprocess

os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/metadata", exist_ok=True)

def download_GTEx(data_dir = "data"):

 urls = [
    "https://storage.googleapis.com/adult-gtex/annotations/v8/metadata-files/GTEx_Analysis_v8_Annotations_SampleAttributesDD.xlsx",
    "https://storage.googleapis.com/adult-gtex/annotations/v8/metadata-files/GTEx_Analysis_v8_Annotations_SubjectPhenotypesDD.xlsx",
    "https://storage.googleapis.com/adult-gtex/annotations/v8/metadata-files/GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt",
    "https://storage.googleapis.com/adult-gtex/annotations/v8/metadata-files/GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt",
    "https://storage.googleapis.com/adult-gtex/bulk-gex/v8/rna-seq/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_reads.gct.gz"
    ]

 for url in urls:
     download = requests.get(url)
     filename = os.path.basename(url)

     if "metadata-files" in url:
        save_dir = os.path.join(data_dir, "metadata")
     else:
        save_dir = os.path.join(data_dir, "raw")

     file_path = os.path.join(save_dir, filename)

     with open(file_path, 'wb') as f:
      f.write(download.content)

     if filename.endswith(".gz"):
            subprocess.run(f"gunzip {file_path}", shell=True, check=True)

if __name__== '__main__':
    download_GTEx()