import os
os.environ["OPENMM_PLUGIN_DIR"] = ""
import numpy as np
#np.typeDict = np.sctypeDict
import MDAnalysis as mda
import warnings
warnings.filterwarnings("ignore", module="MDAnalysis")
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import json
import os
import pandas as pd

# --- Set random seed for reproducibility ---
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# --- Load the best hyperparameters ---
def load_hyperparameters(file_path):
    try:
        with open(file_path, "r") as f:
            best_params = json.load(f)

        if 'hidden_dims' in best_params and isinstance(best_params['hidden_dims'], str):
            best_params['hidden_dims'] = [int(dim) for dim in best_params['hidden_dims'].split('-')]
            best_params['num_layers'] = len(best_params['hidden_dims'])

        best_params.setdefault('epochs', 100)
        best_params.setdefault('beta', 1.0)

        best_val_loss = None  

        return best_params, best_val_loss

    except (json.JSONDecodeError, IndexError, ValueError) as e:
        print(f"Error loading hyperparameters: {e}")
        return None, None


# --- Load hyperparameters ---
best_params, best_val_loss = load_hyperparameters("../step1_hyperparameter_tuning/best_hyperparameters.txt")
if best_params is None:
    print("Failed to load hyperparameters. Exiting...")
    exit()

print(f"Loaded best hyperparameters: {best_params}")

# --- Load trajectory data ---
u = mda.Universe("../protein.top", "../run2.nc")
ca_atoms = u.select_atoms("name CA or name C or name N or name CB or name O")
num_frames = len(u.trajectory)
num_ca_atoms = len(ca_atoms)


# Extract and flatten CA atom coordinates
ca_coords = [ca_atoms.positions.flatten() for ts in u.trajectory]
ca_coords = np.array(ca_coords)
print("Trajectory loaded and coordinates flattened")

# --- Global normalization ---
scaler = MinMaxScaler()
ca_coords_normalized = scaler.fit_transform(ca_coords)

# ✅ Save the scaler parameters in JSON format
scaler_params = {
    "scale": scaler.scale_.tolist(),
    "min": scaler.min_.tolist(),
    "data_min": scaler.data_min_.tolist(),
    "data_max": scaler.data_max_.tolist(),
    "data_range": scaler.data_range_.tolist()
}

with open("scaler_params.json", "w") as f:
    json.dump(scaler_params, f)

print("\n✅ Scaler parameters saved successfully in 'scaler_params.json'")


# --- Split into training and testing sets ---
split_idx = int(0.9 * num_frames)
train_data = ca_coords_normalized[:split_idx]
test_data = ca_coords_normalized[split_idx:]

print(f"Training data shape: {train_data.shape}")
print(f"Testing data shape: {test_data.shape}")


# Convert to PyTorch tensors
train_tensor = torch.tensor(train_data, dtype=torch.float32)
test_tensor = torch.tensor(test_data, dtype=torch.float32)

# Create DataLoader
train_loader = DataLoader(TensorDataset(train_tensor, train_tensor), batch_size=50, shuffle=True)
test_loader = DataLoader(TensorDataset(test_tensor, test_tensor), batch_size=50)


# --- Define the VAE Model ---
class VAE(nn.Module):
    def __init__(self, input_size, latent_size, hidden_dims, num_layers, activation_cls, alpha, dropout_rate):
        super(VAE, self).__init__()

        if activation_cls == "LeakyReLU":
            self.activation = nn.LeakyReLU(alpha)
        else:
            self.activation = getattr(nn, activation_cls)()

        # Encoder
        encoder_layers = []
        in_dim = input_size
        for i in range(num_layers):
            h_dim = hidden_dims[min(i, len(hidden_dims) - 1)]
            encoder_layers.append(nn.Linear(in_dim, h_dim))
            encoder_layers.append(self.activation)
            encoder_layers.append(nn.Dropout(dropout_rate))
            in_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)

        self.fc_mu = nn.Linear(in_dim, latent_size)
        self.fc_logvar = nn.Linear(in_dim, latent_size)

        # Decoder
        decoder_layers = []
        in_dim = latent_size
        for i in range(num_layers):
            h_dim = hidden_dims[min(i, len(hidden_dims) - 1)]
            decoder_layers.append(nn.Linear(in_dim, h_dim))
            decoder_layers.append(self.activation)
            decoder_layers.append(nn.Dropout(dropout_rate))
            in_dim = h_dim
        decoder_layers.append(nn.Linear(in_dim, input_size))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# --- Loss Function with Beta Factor ---
def loss_function(recon_x, x, mu, logvar, beta=1.0):
    MSE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE, beta * KLD


# --- Initialize VAE with best hyperparameters ---
input_size = num_ca_atoms * 3
latent_size = 2

model = VAE(
    input_size=input_size,
    latent_size=latent_size,
    hidden_dims=best_params['hidden_dims'],
    num_layers=best_params['num_layers'],
    activation_cls=best_params['activation_cls'],
    alpha=best_params['alpha'],
    dropout_rate=best_params['dropout_rate']
)

optimizer = torch.optim.Adam(model.parameters(), lr=best_params['learning_rate'], weight_decay=best_params['weight_decay'])

# --- Early Stopping Parameters ---
patience = 10
early_stopping_counter = 0
best_val_loss = float("inf")

# --- Training Loop ---
train_loss_plot, val_loss_plot = [], []
kl_loss_plot, val_kl_loss_plot = [], []
recon_loss_plot, val_recon_loss_plot = [], []

# ✅ Set random seed again for final model training
torch.manual_seed(SEED)
np.random.seed(SEED)

# Initialize lists to store the losses
train_loss_plot, val_loss_plot = [], []
train_kl_loss_plot, val_kl_loss_plot = [], []
train_recon_loss_plot, val_recon_loss_plot = [], []

for epoch in range(best_params['epochs']):
    # --- Training Phase ---
    model.train()
    train_loss, train_recon_loss, train_kl_loss = 0, 0, 0

    for batch, _ in train_loader:
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(batch)

        MSE, KLD = loss_function(recon_batch, batch, mu, logvar, beta=best_params['beta'])
        loss = MSE + KLD

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_recon_loss += MSE.item()
        train_kl_loss += KLD.item()

    # Normalize losses by the dataset size
    train_loss /= len(train_loader.dataset)
    train_recon_loss /= len(train_loader.dataset)
    train_kl_loss /= len(train_loader.dataset)

    # Store training losses
    train_loss_plot.append(train_loss)
    train_recon_loss_plot.append(train_recon_loss)
    train_kl_loss_plot.append(train_kl_loss)

    # --- Validation Phase ---
    model.eval()
    val_loss, val_recon_loss, val_kl_loss = 0, 0, 0

    with torch.no_grad():
        for batch, _ in test_loader:
            recon_batch, mu, logvar = model(batch)

            # ✅ Calculate MSE and KLD separately
            MSE, KLD = loss_function(recon_batch, batch, mu, logvar, beta=best_params['beta'])
            
            # Accumulate the losses
            val_loss += (MSE + KLD).item()
            val_recon_loss += MSE.item()
            val_kl_loss += KLD.item()

    # Normalize validation losses
    val_loss /= len(test_loader.dataset)
    val_recon_loss /= len(test_loader.dataset)
    val_kl_loss /= len(test_loader.dataset)

    # Store validation losses
    val_loss_plot.append(val_loss)
    val_recon_loss_plot.append(val_recon_loss)
    val_kl_loss_plot.append(val_kl_loss)

    # ✅ Display Loss Information
    print(f"Epoch {epoch + 1}/{best_params['epochs']} - "
          f"Train Loss: {train_loss:.4f}, Recon: {train_recon_loss:.4f}, KL: {train_kl_loss:.4f} | "
          f"Val Loss: {val_loss:.4f}, Val Recon: {val_recon_loss:.4f}, Val KL: {val_kl_loss:.4f}")

    # --- Early Stopping Check ---
    min_delta = 1e-4
    if val_loss < best_val_loss - min_delta:
        best_val_loss = val_loss
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1

    if early_stopping_counter >= patience:
        print(f"\nEarly stopping at epoch {epoch + 1}")
        break

# ✅ Save the final model
torch.save(model.state_dict(), "final_vae_model.pth")
print("\nFinal model saved as 'final_vae_model.pth'")

import matplotlib.pyplot as plt

# Set figure size
fig, ax = plt.subplots(1, 3, figsize=(18, 5))

# --- Plot Total Loss ---
ax[0].plot(train_loss_plot, label='Train Loss', color='blue')
ax[0].plot(val_loss_plot, label='Val Loss', color='orange')
ax[0].set_title('Total Loss', fontname='serif', fontweight='bold')

# Labels and Legend
ax[0].set_xlabel('Epochs', fontname='serif', fontweight='bold')
ax[0].set_ylabel('Loss', fontname='serif', fontweight='bold')
ax[0].tick_params(axis='both', labelsize=12)
ax[0].legend(prop={'family': 'serif', 'weight': 'bold'})

# --- Plot KL Loss ---
ax[1].plot(train_kl_loss_plot, label='Train KL Loss', color='green')
ax[1].plot(val_kl_loss_plot, label='Val KL Loss', color='red')
ax[1].set_title('KL Loss', fontname='serif', fontweight='bold')

# Labels and Legend
ax[1].set_xlabel('Epochs', fontname='serif', fontweight='bold')
ax[1].set_ylabel('Loss', fontname='serif', fontweight='bold')
ax[1].tick_params(axis='both', labelsize=12)
ax[1].legend(prop={'family': 'serif', 'weight': 'bold'})

# --- Plot Reconstruction Loss ---
ax[2].plot(train_recon_loss_plot, label='Train Recon Loss', color='purple')
ax[2].plot(val_recon_loss_plot, label='Val Recon Loss', color='brown')
ax[2].set_title('Reconstruction Loss', fontname='serif', fontweight='bold')

# Labels and Legend
ax[2].set_xlabel('Epochs', fontname='serif', fontweight='bold')
ax[2].set_ylabel('Loss', fontname='serif', fontweight='bold')
ax[2].tick_params(axis='both', labelsize=12)
ax[2].legend(prop={'family': 'serif', 'weight': 'bold'})

# ✅ Save the images
plt.tight_layout()
plt.savefig("vae_loss_curves.png", dpi=100, bbox_inches='tight')  # Save all in one image
plt.show()

print("\n✅ Loss plots saved successfully as:")
print(" - vae_loss_curves.png (all losses)")

import pandas as pd

# Initialize lists to store latent representations
train_latent = []
test_latent = []

# Set model to evaluation mode
model.eval()

# Extract latent space (mu) for training and test data
with torch.no_grad():
    # Training set
    for batch, _ in train_loader:
        _, mu, _ = model(batch)
        train_latent.append(mu)

    # Test set
    for batch, _ in test_loader:
        _, mu, _ = model(batch)
        test_latent.append(mu)

# Concatenate latent representations into numpy arrays
train_latent = torch.cat(train_latent, dim=0).cpu().numpy()
test_latent = torch.cat(test_latent, dim=0).cpu().numpy()

# ✅ Ensure the latent vectors are correctly stored in DataFrames
train_latent_df = pd.DataFrame(train_latent, columns=[f'Latent_Dim_{i+1}' for i in range(train_latent.shape[1])])
test_latent_df = pd.DataFrame(test_latent, columns=[f'Latent_Dim_{i+1}' for i in range(test_latent.shape[1])])

# 🚀 Save the latent representations as CSV files
train_latent_df.to_csv('train_latent.csv', index=False)
test_latent_df.to_csv('test_latent.csv', index=False)

# ✅ Confirmation prints
print(f"Train Latent Shape: {train_latent.shape}")
print(f"Test Latent Shape: {test_latent.shape}")
print("CSV files have been saved successfully.")

import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV files
train_data = pd.read_csv('train_latent.csv')
test_data = pd.read_csv('test_latent.csv')

# Create a scatter plot
plt.figure(figsize=(15, 10))

# Plot the train data (Latent_Dim_1 vs Latent_Dim_2)
plt.scatter(train_data['Latent_Dim_1'], train_data['Latent_Dim_2'], color='red', label='Train Set', alpha=0.5, s=50)

# Plot the test data (Latent_Dim_1 vs Latent_Dim_2)
plt.scatter(test_data['Latent_Dim_1'], test_data['Latent_Dim_2'], color='blue', label='Test Set', alpha=0.5, s=50)

# Add labels and title
plt.xlabel('Latent Dimension-1', fontname='serif', fontweight='bold', fontsize=25, labelpad=5)
plt.ylabel('Latent Dimension-2', fontname='serif', fontweight='bold', fontsize=25, labelpad=5)
plt.xticks(fontname='serif', fontweight='bold', fontsize=22)
plt.yticks(fontname='serif', fontweight='bold', fontsize=22)
#plt.title('Latent Space: Train vs Test')
plt.gca().spines["top"].set_linewidth(2.5)
plt.gca().spines["right"].set_linewidth(2.5)
plt.gca().spines["left"].set_linewidth(2.5)
plt.gca().spines["bottom"].set_linewidth(2.5)
plt.tick_params(axis='both', which='major', labelsize=15, length=8, width=2.5)
# Add legend
plt.legend(fontsize=18, fancybox=True, frameon=True, loc='best', edgecolor='black')
plt.grid()
# Show the plot
plt.savefig("latent_space.png", dpi=150, bbox_inches='tight')
