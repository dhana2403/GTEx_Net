import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import numpy as np

# ---- Conditional Decoder ----
class ConditionalDecoder(nn.Module):
    def __init__(self, latent_dim, num_batches, output_dim):
        super().__init__()
        self.batch_embed = nn.Embedding(num_batches, 8)  # learnable batch embedding
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 8, 8),
            nn.BatchNorm1d(8),
            nn.Tanh(),
            nn.Linear(8, output_dim)
        )

    def forward(self, z, batch_idx):
        b_emb = self.batch_embed(batch_idx)
        z_cat = torch.cat([z, b_emb], dim=1)
        return self.net(z_cat)

# ---- Autoencoder with Conditional Decoder & Tissue Classifier ----
class AutoencoderWithClassifier(nn.Module):
    def __init__(self, input_dim, latent_dim, num_tissues, num_batches, dropout_prob=0.1):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.BatchNorm1d(8),
            nn.Tanh(),
            nn.Dropout(dropout_prob),
            nn.Linear(8, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
        )
        # Conditional Decoder
        self.decoder = ConditionalDecoder(latent_dim, num_batches, input_dim)
        # Classifier head on latent
        self.classifier = nn.Linear(latent_dim, num_tissues)

    def forward(self, x, batch_idx):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent, batch_idx)
        tissue_logits = self.classifier(latent)
        return reconstructed, latent, tissue_logits

# ---- Data scaling ----
def scale_data(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

# ---- Training ----
def train_autoencoder_supervised(
    X, tissue_labels, batch_ids, batch_size=64, epochs=50, lr=1e-3, alpha=1.0, device='cpu'
):
    X_scaled, scaler = scale_data(X)

    input_dim = X.shape[1]
    unique_tissues = np.unique(tissue_labels)
    tissue_to_idx = {t: i for i, t in enumerate(unique_tissues)}
    y_indices = np.array([tissue_to_idx[t] for t in tissue_labels])

    unique_batches = np.unique(batch_ids)
    batch_to_idx = {b: i for i, b in enumerate(unique_batches)}
    b_indices = np.array([batch_to_idx[b] for b in batch_ids])

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_indices, dtype=torch.long).to(device)
    b_tensor = torch.tensor(b_indices, dtype=torch.long).to(device)

    dataset = TensorDataset(X_tensor, y_tensor, b_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = AutoencoderWithClassifier(
        input_dim=input_dim,
        latent_dim=4,
        num_tissues=len(unique_tissues),
        num_batches=len(unique_batches)
    ).to(device)

    criterion_recon = nn.MSELoss()
    criterion_class = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    model.train()
    for epoch in range(epochs):
        total_loss = total_recon = total_class = 0
        for x_batch, y_batch, b_batch in loader:
            optimizer.zero_grad()
            x_recon, latent, tissue_logits = model(x_batch, b_batch)

            loss_recon = criterion_recon(x_recon, x_batch)
            loss_class = criterion_class(tissue_logits, y_batch)
            loss = loss_recon + alpha * loss_class

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += loss_recon.item()
            total_class += loss_class.item()

        print(f"Epoch {epoch+1}/{epochs} | Total: {total_loss/len(loader):.4f} | "
              f"Recon: {total_recon/len(loader):.4f} | Class: {total_class/len(loader):.4f}")

    return model, scaler

# ---- Get latent representation ----
def get_latent_representation(model, X, batch_indices, scaler, device='cpu'):
    model.eval()
    X_scaled = scaler.transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    b_tensor = torch.tensor(batch_indices, dtype=torch.long).to(device)
    with torch.no_grad():
        _, latent, _ = model(X_tensor, b_tensor)
    return latent.cpu().numpy()
