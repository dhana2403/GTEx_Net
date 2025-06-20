# GTEx_Net: Gene Expression Analysis & Batch Correction Pipeline

This repository provides a streamlined pipeline for preprocessing, normalizing, and batch-correcting RNA-seq expression data from the [GTEx project](https://gtexportal.org/home/), enabling downstream analyses such as clustering and visualization.

---

### Features

- Preprocessing of GTEx metadata and raw read count files  
- Log2-CPM normalization per tissue  
- Autoencoder-based batch correction  
- PCA / 3D visualization support  
- Tissue-wise file outputs for modular analysis  

---

### Project Structure

GTEx_Net/
│
├── data/
│ ├── download.py # Downloads raw data and metadata files from GTEx
│
├── scripts/
│ ├── prepare_data.py # End-to-end preparation pipeline
│ ├── run_batch_correction.py # Trains autoencoder and saves latent outputs
│ └── visualize_pca.py # 3D PCA visualizations
├  src/
│   ├── data_preprocessing.py # Metadata parsing, filtering
│   ├── data_normalization.py # Log2CPM transformation
│   └── autoencoder.py # Autoencoder model, training, scaling
│
└── README.md

### Getting Started

#### 1. Install dependencies

```bash
pip install numpy pandas torch scikit-learn matplotlib seaborn

#### 2. Prepare raw data
Download the following GTEx files and place them accordingly:

GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt → data/metadata/

GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt → data/metadata/

GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_reads.gct → data/raw/

#### 3. Run data preparation

bash
python scripts/prepare_data.py
4. Run batch correction
bash
python scripts/run_batch_correction.py
5. Visualize in 3D PCA
bash
python scripts/visualize_pca.py

#### Method Summary

Normalization: log2(CPM + 1), per tissue

Batch Correction: Autoencoder trained per tissue, latent space is used as corrected expression

Latent Space: 10-dimensional compressed representation of gene expression

Visualization: PCA or 3D PCA on corrected data
