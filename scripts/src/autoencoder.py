import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import numpy as np

#COMPLETE CODE WILL BE MADE AVAILABLE SOON#
    
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        tissue_logits = self.classifier(latent)
        return reconstructed, latent, tissue_logits

def scale_data(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

def train_autoencoder_supervised(X, tissue_labels, batch_ids, batch_size=64, epochs=50, lr=1e-3, alpha=1.0, device='cpu'):
    X_scaled, scaler = scale_data(X)
    input_dim = X.shape[1]
    output_dim = input_dim
    unique_tissues = np.unique(tissue_labels)
    tissue_to_idx = {t:i for i,t in enumerate(unique_tissues)}
    y_indices = np.array([tissue_to_idx[t] for t in tissue_labels])

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_indices, dtype=torch.long).to(device)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = AutoencoderWithClassifier(input_dim, output_dim, latent_dim=10, num_tissues=len(unique_tissues)).to(device)
    criterion_recon = nn.MSELoss()
    criterion_class = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        total_recon = 0
        total_class = 0
        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            x_recon, latent, tissue_logits = model(x_batch)
            loss_recon = criterion_recon(x_recon, x_batch)
            loss_class = criterion_class(tissue_logits, y_batch)
            loss = loss_recon + alpha * loss_class
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += loss_recon.item()
            total_class += loss_class.item()

        print(f"Epoch {epoch+1}/{epochs}, Total Loss: {total_loss/len(loader):.4f}, Recon: {total_recon/len(loader):.4f}, Class: {total_class/len(loader):.4f}")

    return model, scaler, tissue_to_idx, None

def get_latent_representation(model, X, scaler, device='cpu'):
    model.eval()
    X_scaled = scaler.transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    with torch.no_grad():
        _, latent, _ = model(X_tensor)
    return latent.cpu().numpy()

    
    
