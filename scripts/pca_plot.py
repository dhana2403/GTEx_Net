import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pickle

# ========== Configuration ==========
INPUT_DIR = "./data/processed/expression/readcounts_tmm_all/"
OUTPUT_DIR = "./data/processed/expression/autoencoder/"
METADATA_PATH = "./data/processed/attphe.pkl"
TOP_GENES = 5000
LATENT_DIM = 8
MIN_SAMPLES = 10
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== Load Metadata ==========
with open(METADATA_PATH, 'rb') as f:
    metadata = pickle.load(f)

# ========== Model Definition ==========
def build_autoencoder(input_dim):
    input_layer = keras.Input(shape=(input_dim,))
    x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001))(input_layer)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation='relu')(x)
    latent = layers.Dense(LATENT_DIM, activation='relu', name='latent')(x)
    x = layers.Dense(32, activation='relu')(latent)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation='relu')(x)
    output = layers.Dense(input_dim, activation='linear')(x)

    autoencoder = keras.Model(inputs=input_layer, outputs=output)
    autoencoder.compile(optimizer='adam', loss='mse')

    encoder = keras.Model(inputs=input_layer, outputs=latent)
    return autoencoder, encoder

# ========== Tissue Processor ==========
def process_tissue(tissue_file):
    tissue_name = os.path.basename(tissue_file).replace(".pkl", "")
    print(f"\n📦 Processing {tissue_name}")

    with open(tissue_file, 'rb') as f:
        expr = pickle.load(f)

    sample_ids = expr.columns
    attr = metadata[metadata['samp_id'].isin(sample_ids)].copy()

    if expr.shape[1] < MIN_SAMPLES:
        print(f"❌ Skipping {tissue_name}: only {expr.shape[1]} samples")
        return

    # Ensure metadata is ordered same as expr columns (samples)
    attr = attr.set_index('samp_id').loc[sample_ids]

    # Filter top variable genes
    gene_vars = expr.var(axis=1)
    top_genes = gene_vars.sort_values(ascending=False).head(TOP_GENES).index
    expr = expr.loc[top_genes]

    # If sample size > 100, use autoencoder
    if expr.shape[1] > 100:
        print("Using autoencoder-based batch correction")

        # Normalize across samples
        scaler = StandardScaler()
        X = scaler.fit_transform(expr.T)  # samples × genes

        # Build & train autoencoder
        autoencoder, encoder = build_autoencoder(X.shape[1])
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        history = autoencoder.fit(
            X, X,
            epochs=100,
            batch_size=32,
            validation_split=0.1,
            callbacks=[early_stop],
            verbose=0
        )

        # Plot loss
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.legend()
        plt.title(f"Loss - {tissue_name}")
        plt.savefig(os.path.join(OUTPUT_DIR, f"{tissue_name}_loss.png"))
        plt.show()

        # ✅ Use autoencoder output as corrected expression
        X_denoised = autoencoder.predict(X)
        adjusted = pd.DataFrame(X_denoised, index=sample_ids, columns=expr.index).T

    else:
        print("⚠️ Not enough samples for autoencoder — saving uncorrected expression")
        adjusted = expr  # no correction applied

    output_file = os.path.join(OUTPUT_DIR, f"{tissue_name}.pkl")
    with open(output_file, 'wb') as f:
        pickle.dump(adjusted, f)

    print(f"✅ Saved adjusted data: {output_file} — shape: {adjusted.shape}")

# ========== Run all tissues ==========
for filename in os.listdir(INPUT_DIR):
    if filename.endswith(".pkl"):
        tissue_file = os.path.join(INPUT_DIR, filename)
        process_tissue(tissue_file)