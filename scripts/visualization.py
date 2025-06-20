import os
import pickle
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_corrected_data(input_dir):
    all_data = []
    labels = []

    for filename in sorted(os.listdir(input_dir)):
        if filename.endswith("_corrected.pkl"):
            tissue = filename.replace("_corrected.pkl", "")
            filepath = os.path.join(input_dir, filename)

            df = pickle.load(open(filepath, 'rb'))  # samples x latent dim
            all_data.append(df.values)
            labels.extend([tissue] * df.shape[0])
    
    X = np.vstack(all_data)
    return X, labels

def plot_pca_3d(X, labels, title="3D PCA of Batch-Corrected Data"):
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    tissues = sorted(set(labels))
    colors = plt.cm.tab20(np.linspace(0, 1, len(tissues)))

    for tissue, color in zip(tissues, colors):
        indices = [i for i, l in enumerate(labels) if l == tissue]
        ax.scatter(X_pca[indices, 0], X_pca[indices, 1], X_pca[indices, 2],
                   label=tissue, s=30, alpha=0.7, color=color)

    ax.set_title(title)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)")
    ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]*100:.2f}%)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    corrected_dir = "/Users/dhanalakshmijothi/Desktop/python/GTEx_Net/scripts/data/processed/batch_corrected"
    X, labels = load_corrected_data(corrected_dir)
    plot_pca_3d(X, labels)
