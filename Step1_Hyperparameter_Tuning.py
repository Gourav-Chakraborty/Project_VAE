import os
os.environ["OPENMM_PLUGIN_DIR"] = ""
import MDAnalysis as mda
import warnings
warnings.filterwarnings("ignore", module="MDAnalysis")
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import optuna
import joblib
import random
import json
import pandas as pd
import plotly.io as pio
import optuna.visualization as vis
import matplotlib.pyplot as plt

# ----------------------------
# ✅ 1️⃣ Set Random Seed for Reproducibility
# ----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

os.environ['PYTHONHASHSEED'] = str(SEED)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ----------------------------
# ✅ 2️⃣ Load MD Trajectory Data
# ----------------------------
print("🔹 Loading trajectory...")
u = mda.Universe("../protein.top", "../run2.nc")
ca_atoms = u.select_atoms("name CA or name C or name N or name CB or name O")
num_frames = len(u.trajectory)

# Extract coordinates
ca_coords = np.array([ts.positions.flatten() for ts in u.trajectory])
print(f"🔹 Trajectory loaded with {num_frames} frames and {ca_coords.shape[1]} features.")

# ----------------------------
# ✅ 3️⃣ Preprocessing
# ----------------------------
scaler = MinMaxScaler()
ca_coords_normalized = scaler.fit_transform(ca_coords)


split_idx = int(0.90 * num_frames)
train_data = ca_coords_normalized[:split_idx]
test_data = ca_coords_normalized[split_idx:]
print(train_data.shape)
print(test_data.shape)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔹 Using device: {device}")

train_tensor = torch.tensor(train_data, dtype=torch.float32).to(device)
test_tensor = torch.tensor(test_data, dtype=torch.float32).to(device)

batch_size = 50
train_loader = DataLoader(TensorDataset(train_tensor, train_tensor), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(test_tensor, test_tensor), batch_size=batch_size)

# ----------------------------
# ✅ 4️⃣ VAE Model Definition
# ----------------------------
class VAE(nn.Module):
    def __init__(self, input_size, latent_size, hidden_dims, activation_cls, alpha, dropout_rate):
        super(VAE, self).__init__()
        self.activation = activation_cls(alpha) if activation_cls == nn.LeakyReLU else activation_cls()

        # Encoder
        encoder_layers = []
        in_dim = input_size
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, h_dim))
            encoder_layers.append(self.activation)
            encoder_layers.append(nn.Dropout(dropout_rate))
            in_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)

        self.fc_mu = nn.Linear(in_dim, latent_size)
        self.fc_logvar = nn.Linear(in_dim, latent_size)

        # Decoder
        decoder_layers = []
        for h_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(latent_size if len(decoder_layers) == 0 else in_dim, h_dim))
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

# ----------------------------
# ✅ 5️⃣ Loss Function
# ----------------------------
def loss_function(recon_x, x, mu, logvar):
    MSE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE, KLD

# ----------------------------
# ✅ 6️⃣ Optuna Objective Function
# ----------------------------
def objective(trial):
    hidden_dims = list(map(int, trial.suggest_categorical(
        'hidden_dims', ['256-128-64', '128-128-128', '256-128-128-64', '512-256-128-64','512-256-256-128-64', '1024-512-512-256-256-128']).split('-')))

    activation_mapping = {
        "ReLU": nn.ReLU,
        "LeakyReLU": nn.LeakyReLU,
        "SiLU": nn.SiLU,
        "GELU": nn.GELU
    }
    activation_cls = activation_mapping[trial.suggest_categorical('activation_cls', list(activation_mapping.keys()))]

    alpha = trial.suggest_float('alpha', 1e-5, 1e-3, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.005, 0.2)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)

    beta = trial.suggest_float('beta', 1e-5, 1e-1, log=True)
    epochs = trial.suggest_int('epochs', 50, 400)

    patience = 10
    min_delta = 1e-4  

    optimizer_choice = trial.suggest_categorical('optimizer', ['Adam', 'SGD'])
    optimizer_cls = optim.Adam if optimizer_choice == "Adam" else optim.SGD

    input_size = train_tensor.shape[1]
    latent_size = 2

    model = VAE(input_size, latent_size, hidden_dims, activation_cls, alpha, dropout_rate).to(device)
    optimizer = optimizer_cls(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        for batch, _ in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            recon_batch, mu, logvar = model(batch)
            MSE, KLD = loss_function(recon_batch, batch, mu, logvar)

            loss = MSE + beta * KLD
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for val_batch, _ in test_loader:
                val_batch = val_batch.to(device)
                recon_batch, mu, logvar = model(val_batch)
                MSE, KLD = loss_function(recon_batch, val_batch, mu, logvar)
                val_loss += (MSE + beta * KLD).item()

        val_loss /= len(test_loader)

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            epochs_no_improve = 0  
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"⏹️ Early stopping at epoch {epoch + 1}.")
            break

    return best_val_loss

# ----------------------------
# ✅ 7️⃣ Optuna Study Execution
# ----------------------------
study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=125)

# ✅ Save Trials as CSV
trials_df = study.trials_dataframe()
trials_df.to_csv("optuna_trials.csv", index=False)

# ✅ Save Best Hyperparameters
best_params = study.best_params
with open("best_hyperparameters.txt", "w") as f:
    json.dump(best_params, f, indent=4)

# ✅ Visualization with saving to files
optimization_history = vis.plot_optimization_history(study)
param_importances = vis.plot_param_importances(study)
slice_plot = vis.plot_slice(study)

#optimization_history.write_html("optimization_history.html")
#param_importances.write_html("param_importances.html")
#slice_plot.write_html("slice_plot.html")

# Save as PNG images
optimization_history.write_image("optimization_history.png")
param_importances.write_image("param_importances.png")
slice_plot.write_image("slice_plot.png")

print("✅ Visualization saved as HTML and PNG files.")

print("✅ Trials, best hyperparameters, and visualizations saved successfully!")

