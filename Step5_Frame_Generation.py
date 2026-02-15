import os
os.environ["OPENMM_PLUGIN_DIR"] = ""
import numpy as np
import MDAnalysis as mda
import warnings
warnings.filterwarnings("ignore", module="MDAnalysis")
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import json# --- Set random seed for reproducibility ---
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# --- Load hyperparameters ---
def load_hyperparameters(file_path):
    try:
        with open(file_path, "r") as f:
            best_params = json.load(f)

        if isinstance(best_params.get("hidden_dims"), str):
            best_params["hidden_dims"] = [int(dim) for dim in best_params["hidden_dims"].split('-')]
            best_params["num_layers"] = len(best_params["hidden_dims"])

        # Set defaults if not found
        best_params["epochs"] = best_params.get("epochs", 100)
        best_params["beta"] = best_params.get("beta", 1.0)

        return best_params

    except Exception as e:
        print(f"❌ Failed to load hyperparameters: {e}")
        return None


# ✅ Load hyperparameters
best_params = load_hyperparameters("../../step1_hyperparameter_tuning/best_hyperparameters.txt")
if best_params is None:
    exit()
print(f"✅ Loaded best hyperparameters: {best_params}")

# --- Load scaler ---
def load_scaler(scaler_file):
    with open(scaler_file, "r") as f:
        params = json.load(f)

    scaler = MinMaxScaler()
    scaler.scale_ = np.array(params["scale"])
    scaler.min_ = np.array(params["min"])
    scaler.data_min_ = np.array(params["data_min"])
    scaler.data_max_ = np.array(params["data_max"])
    scaler.data_range_ = np.array(params["data_range"])
    scaler.n_features_in_ = len(scaler.scale_)
    return scaler


scaler = load_scaler("../../step2_model_building/scaler_params.json")
print("✅ Scaler loaded successfully")
import os
# --- Load MD trajectory ---
u = mda.Universe("../../protein.top", "../../run2.nc")
ca_atoms = u.select_atoms("name CA or name C or name N or name CB or name O")
num_ca_atoms = len(ca_atoms)
num_frames = len(u.trajectory)

# Use memory-efficient generator approach
def extract_normalized_coordinates(universe, atom_group, scaler):
    for ts in universe.trajectory:
        yield scaler.transform(atom_group.positions.flatten()[np.newaxis, :])[0]

# Preallocate and fill normalized coordinates (efficient)
ca_coords_normalized = np.empty((num_frames, num_ca_atoms * 3), dtype=np.float32)
for i, normalized_coords in enumerate(extract_normalized_coordinates(u, ca_atoms, scaler)):
    ca_coords_normalized[i] = normalized_coords

# ✅ Convert to PyTorch tensor
data_tensor = torch.tensor(ca_coords_normalized, dtype=torch.float32)
print("✅ Trajectory normalized and converted to tensor")
# --- VAE Architecture ---
class VAE(nn.Module):
    def __init__(self, input_size, latent_size, hidden_dims, num_layers, activation_cls, alpha, dropout_rate):
        super(VAE, self).__init__()
        act_fn = nn.LeakyReLU(alpha) if activation_cls == "LeakyReLU" else getattr(nn, activation_cls)()

        # Encoder
        encoder = []
        in_dim = input_size
        for i in range(num_layers):
            h_dim = hidden_dims[min(i, len(hidden_dims) - 1)]
            encoder += [nn.Linear(in_dim, h_dim), act_fn, nn.Dropout(dropout_rate)]
            in_dim = h_dim
        self.encoder = nn.Sequential(*encoder)
        self.fc_mu = nn.Linear(in_dim, latent_size)
        self.fc_logvar = nn.Linear(in_dim, latent_size)

        # Decoder
        decoder = []
        in_dim = latent_size
        for i in range(num_layers):
            h_dim = hidden_dims[min(i, len(hidden_dims) - 1)]
            decoder += [nn.Linear(in_dim, h_dim), act_fn, nn.Dropout(dropout_rate)]
            in_dim = h_dim
        decoder.append(nn.Linear(in_dim, input_size))
        self.decoder = nn.Sequential(*decoder)

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


# ✅ Initialize and load model
input_size = num_ca_atoms * 3
latent_size = 2
model = VAE(
    input_size=input_size,
    latent_size=latent_size,
    hidden_dims=best_params["hidden_dims"],
    num_layers=best_params["num_layers"],
    activation_cls=best_params["activation_cls"],
    alpha=best_params["alpha"],
    dropout_rate=best_params["dropout_rate"]
)

model.load_state_dict(torch.load("../../step2_model_building/final_vae_model.pth"))
model.eval()
print("✅ Model loaded and ready for inference.")

# --- Select atoms of interest (N, CA, CB, C, O) ---
atoms_of_interest = u.select_atoms("name N or name CA or name CB or name C or name O")

# Group atoms by residue ID and sort each residue group by atom name (N, CA, CB, C, O)
sorted_atoms = []
for resid in np.unique(atoms_of_interest.resids):
    residue_atoms = atoms_of_interest.select_atoms(f"resid {resid}")
    sorted_atoms.extend(
        sorted(residue_atoms, key=lambda atom: ["N", "CA", "CB", "C", "O"].index(atom.name))
    )

sorted_atoms = u.atoms[np.array([atom.index for atom in sorted_atoms])]

# --- User Input ---
print("Please enter a point in the latent space (e.g., x y):")
latent_point_input = input("Enter the point (x y): ").strip().split()
output_name = input("Enter output file name (without .pdb extension): ").strip()

try:
    latent_point = np.array([float(coord) for coord in latent_point_input]).reshape(1, -1)
except ValueError:
    print("❌ Invalid input. Please enter numeric values for the point.")
    exit()

# --- Decode and Save Structure ---
output_dir = "decoded_structures"
os.makedirs(output_dir, exist_ok=True)

latent_tensor = torch.tensor(latent_point, dtype=torch.float32)

with torch.no_grad():
    decoded_vector = model.decode(latent_tensor).cpu().numpy()

# Reverse scaling to recover original coordinates
decoded_vector_original = scaler.inverse_transform(decoded_vector).reshape(-1, len(sorted_atoms), 3)

# Assign decoded coordinates to sorted atoms
sorted_atoms.positions = decoded_vector_original[0]

# Save structure to PDB file
output_file_path = os.path.join(output_dir, f"{output_name}.pdb")
sorted_atoms.write(output_file_path)
print(f"✅ Decoded structure saved to '{output_file_path}'")


