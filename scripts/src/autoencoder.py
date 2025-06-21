import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import numpy as np

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=3):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8),
            nn.ReLU(),
            nn.Linear(8, input_dim)
        )
    
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent
    
def scale_data(X):
         """
         Standard scale the data (mean=0, std=1) per feature
          Input: numpy array (samples x features)
         Output: scaled numpy array, scaler object
         """
         scaler = StandardScaler()
         X_scaled = scaler.fit_transform(X)
         return X_scaled, scaler
    
def train_autoencoder(X, batch_size=64, epochs=50, lr=1e-3, device='cpu'):
        """
         Train autoencoder on given data X
         Inputs:
         X - numpy array, samples x features
         batch_size - batch size for training
         epochs - number of epochs
         lr - learning rate
         device - 'cpu' or 'cuda'
         Returns:
         trained model
         scaler used for data scaling
          """
        X_scaled, scaler = scale_data(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model = Autoencoder(input_dim=X.shape[1]).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in loader:
                x = batch[0]
                optimizer.zero_grad()
                x_recon, _ = model(x)
                loss = criterion(x_recon, x)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss/len(loader)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

        return model, scaler
    
def get_latent_representation(model, X, scaler, device='cpu'):
        """
         Generate latent representations for data X using trained model and scaler
         Inputs:
         model - trained Autoencoder model
         X - numpy array samples x features
         scaler - fitted StandardScaler
         device - 'cpu' or 'cuda'
         Returns:
         latent numpy array (samples x latent_dim)
         """
        model.eval()
        X_scaled = scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
        with torch.no_grad():
         _, latent = model(X_tensor)
        return latent.cpu().numpy()
    

    
    
