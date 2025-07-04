# GTEx_Net: Batch Correction Pipeline using Supervised Autoencoder

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

<pre lang="markdown"> ### ``` GTEx_Net/ ├── data/ │ └── download.py # Downloads raw data and metadata files from GTEx │ ├── scripts/ │ ├── prepare_data.py # End-to-end preparation pipeline │ ├── run_batch_correction.py # Trains autoencoder and batch corrected outputs │ ├── visualize_pca.py # 3D PCA visualizations │ ├── src/ │ ├── data_preprocessing.py # Metadata cleaning, filtering │ ├── data_normalization.py # Log2CPM transformation │ └── autoencoder.py # Autoencoder model, training, scaling │ └── README.md ``` </pre>

### Neural Network Architecture

<p align="center">
  <img src="https://raw.githubusercontent.com/dhana2403/GTEx_Net/main/neural_network.drawio.png" width="600">
</p>

### Getting Started

#### 1. Install dependencies

pip install numpy pandas torch scikit-learn matplotlib seaborn

#### 2. Prepare raw data

Download the following GTEx files and place them accordingly:

GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt → data/metadata/

GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt → data/metadata/

GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_reads.gct → data/raw/

#### 3. Run data preparation

python scripts/prepare_data.py

#### 4. Run batch correction

python scripts/run_batch_correction.py

#### 5. Visualize in 3D PCA

python scripts/visualize_pca.py

#### Method Summary

Normalization: log2(CPM + 1), per tissue

Batch Correction: Supervised autoencoder trained per tissue, using tissue labels to guide the latent space learning; the latent space representation serves as batch-corrected expression

Latent Space: 3-dimensional compressed representation of gene expression

Visualization: PCA or 3D PCA on corrected data

#### Results

[View the 3D PCA plot before batch correction](https://dhana2403.github.io/3D_plots/3d_pca_plot_all_tissues_cpm.html)
[View the 3D PCA plot after batch correction](https://dhana2403.github.io/3D_plots/3d_pca_plot_all_tissues_after_batchcorrection_autoencoder)


