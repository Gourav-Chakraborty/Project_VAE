import os
os.environ["OPENMM_PLUGIN_DIR"] = ""
import numpy as np
import MDAnalysis as mda
import warnings
warnings.filterwarnings("ignore", module="MDAnalysis")
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from MDAnalysis.analysis.rms import RMSD
import json
import os

# --- Set random seed for reproducibility ---
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
best_params = load_hyperparameters("../step1_hyperparameter_tuning/best_hyperparameters.txt")
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


scaler = load_scaler("../step2_model_building/scaler_params.json")
print("✅ Scaler loaded successfully")

# --- Load MD trajectory ---
u = mda.Universe("../protein.top", "../run2.nc")
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

model.load_state_dict(torch.load("../step2_model_building/final_vae_model.pth"))
model.eval()
print("✅ Model loaded and ready for inference.")

import pandas as pd


# --- Load the latent space values from the CSV ---
latent_csv = "../step2_model_building/test_latent.csv"  # Path to your test latent CSV file
latent_data = pd.read_csv(latent_csv)

# --- Select atoms of interest (N, CA, CB, C, O) ---
atoms_of_interest = u.select_atoms("name N or name CA or name CB or name C or name O")

# Group atoms by residue ID and sort each residue group by atom name (N, CA, CB, C, O)
sorted_atoms = []
for resid in np.unique(atoms_of_interest.resids):
    residue_atoms = atoms_of_interest.select_atoms(f"resid {resid}")
    sorted_atoms.extend(
        sorted(residue_atoms, key=lambda atom: ["N", "CA", "CB", "C", "O"].index(atom.name))
    )

# Convert the sorted_atoms list back to an AtomGroup
sorted_atoms = u.atoms[np.array([atom.index for atom in sorted_atoms])]

# --- Create the output directory ---
output_dir = "test_set"
os.makedirs(output_dir, exist_ok=True)

# --- Iterate over every 200th frame in the latent space CSV ---
for frame_idx, latent_row in enumerate(latent_data.itertuples(index=False), start=1):
    if (frame_idx - 1) % 200 == 0:  # Process every 200th frame
        latent_vector = np.array(latent_row).reshape(1, -1)  # Read the latent vector

        # Decode the latent vector
        latent_tensor = torch.tensor(latent_vector, dtype=torch.float32)
        model.eval()  # Ensure model is in evaluation mode

        with torch.no_grad():  # Disable gradient computation
            decoded_vector = model.decode(latent_tensor)

        # Convert the decoded vector back to numpy
        decoded_vector = decoded_vector.cpu().numpy()

        # Reverse the MinMax scaling to recover the original coordinates
        try:
            decoded_vector_original = scaler.inverse_transform(decoded_vector)
        except Exception as e:
            print(f"Error in inverse transforming the data for frame {frame_idx}: {e}")
            break

        # Reshape the decoded coordinates back to the original shape (frames, atoms, 3)
        decoded_vector_original = decoded_vector_original.reshape(-1, len(sorted_atoms), 3)

        # Get the decoded coordinates for this frame
        decoded_coords = decoded_vector_original[0]  # First frame in this latent vector

        # Assign the decoded coordinates to the sorted atoms
        sorted_atoms.positions = decoded_coords

        # Save the updated structure to a PDB file
        output_file = os.path.join(output_dir, f"test_frame{frame_idx}.pdb")
        sorted_atoms.write(output_file)

        print(f"Decoded structure for test_frame{frame_idx} saved to '{output_file}'.")


import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import MDAnalysis as mda
from pdbfixer import PDBFixer
from openmm.app import PDBFile, Modeller, ForceField, Simulation, NoCutoff
from openmm import LangevinIntegrator, unit
from openmm.app import PME, HBonds

# --- Parameters ---
pdb_dir = "test_set"
forcefield = ForceField("amber14-all.xml", "amber14/tip3p.xml")
temp = 310 * unit.kelvin
step_count = 1000
selection_str = "name N or name CA or name CB or name C or name O"

# --- Minimize structures ---
pdb_files = sorted(
    [f for f in os.listdir(pdb_dir) if f.endswith(".pdb")],
    key=lambda x: int(x.split("test_frame")[1].split(".pdb")[0])
)

for pdb_name in tqdm(pdb_files, desc="Minimizing PDBs"):
    print(f"\nMinimizing: {pdb_name}")
    pdb_path = os.path.join(pdb_dir, pdb_name)

    # Fix with PDBFixer
    fixer = PDBFixer(filename=pdb_path)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(pH=7.5)

    fixed_pdb_path = pdb_path  # Overwrite the same file
    with open(fixed_pdb_path, 'w') as out_file:
        PDBFile.writeFile(fixer.topology, fixer.positions, out_file)

    # Now proceed with minimization using OpenMM
    pdb = PDBFile(fixed_pdb_path)
    modeller = Modeller(pdb.topology, pdb.positions)

    system = forcefield.createSystem(modeller.topology, nonbondedMethod=NoCutoff, constraints=HBonds)
    integrator = LangevinIntegrator(temp, 1.0/unit.picoseconds, 0.002*unit.picoseconds)
    simulation = Simulation(modeller.topology, system, integrator)
    simulation.context.setPositions(modeller.positions)

    simulation.minimizeEnergy(maxIterations=step_count)
    state = simulation.context.getState(getPositions=True)
    positions = state.getPositions()

    with open(fixed_pdb_path, 'w') as f:
        PDBFile.writeFile(simulation.topology, positions, f)

print("\n✅ All structures minimized and overwritten with fixed versions.")


# --- RMSD Calculation against first frame of trajectory ---
# Load the original trajectory
u = mda.Universe("../protein.top", "../run2.nc")  # Adjust as needed
num_frames = len(u.trajectory)
split_idx = int(0.9 * num_frames)
test_start_frame = split_idx + 1  # e.g., 72001 if total is 80000

# Reference: first frame
u.trajectory[0]
ref_atoms = u.select_atoms(selection_str)
ref_coords = ref_atoms.positions.copy()

# Prepare RMSD output
rmsd_records = []

for pdb_name in tqdm(pdb_files, desc="Calculating RMSDs"):
    decoded_idx = int(pdb_name.split("test_frame")[1].split(".pdb")[0])
    actual_idx = test_start_frame + decoded_idx - 1

    if actual_idx >= num_frames:
        continue

    # Load decoded structure
    decoded_path = os.path.join(pdb_dir, pdb_name)
    decoded_universe = mda.Universe(decoded_path)
    decoded_atoms = decoded_universe.select_atoms(selection_str)

    # Load actual frame from trajectory
    u.trajectory[actual_idx - 1]
    actual_atoms = u.select_atoms(selection_str)

    # RMSD calculations
    rmsd_decoded = np.sqrt(np.mean(np.sum((decoded_atoms.positions - ref_coords) ** 2, axis=1)))
    rmsd_actual = np.sqrt(np.mean(np.sum((actual_atoms.positions - ref_coords) ** 2, axis=1)))

    rmsd_records.append([
        actual_idx, rmsd_actual, decoded_idx, rmsd_decoded
    ])

# Save to CSV
rmsd_df = pd.DataFrame(rmsd_records, columns=["Actual_Frame", "RMSD_Actual", "Decoded_PDB", "RMSD_Decoded"])
rmsd_df.to_csv("rmsd_comparison.csv", index=False)
print("\n📊 RMSD comparison complete. Results saved to 'rmsd_comparison.csv'")


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# Load the RMSD results
rmsd_df = pd.read_csv("rmsd_comparison.csv")

# Extract columns for correlation
x = rmsd_df["RMSD_Actual"]
y = rmsd_df["RMSD_Decoded"]

# Compute Pearson correlation
corr_coef, p_value = pearsonr(x, y)

# Plot
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")
sns.regplot(
    x=x,
    y=y,
    ci=None,
    scatter_kws={'s': 80, 'alpha': 0.7},  # size 80 ~ marker size 20
    line_kws={'color': 'darkred', 'linewidth': 3.0}
)

plt.xlabel("RMSD (Actual, Å)", fontsize=18, fontweight='bold', fontname='serif', labelpad=10)
plt.ylabel("RMSD (Predicted, Å)", fontsize=18, fontweight='bold', fontname='serif', labelpad=10)

# Customize tick parameters
plt.tick_params(axis='both', which='major', length=8, width=2.5, labelsize=15)
plt.tick_params(axis='both', which='minor', length=8, width=2.5)

# Increase border (spine) thickness
for spine in plt.gca().spines.values():
    spine.set_linewidth(2)

plt.tight_layout()
plt.savefig("rmsd_actual_vs_predicted.png", dpi=150)

