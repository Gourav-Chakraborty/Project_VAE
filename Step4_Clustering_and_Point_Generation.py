import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from scipy.stats import gaussian_kde
from matplotlib.ticker import ScalarFormatter, FuncFormatter
from matplotlib.font_manager import FontProperties
import matplotlib
import re  # Regular expression library
from skopt import gp_minimize
from skopt.space import Real
from sklearn.metrics import silhouette_score  # Import Silhouette score
from sklearn.preprocessing import StandardScaler  # Import StandardScaler

# --- Load Data ---
train_df = pd.read_csv("train_latent.csv")
test_df = pd.read_csv("test_latent.csv")
combined_df = pd.concat([train_df, test_df], ignore_index=True)
latent_space = combined_df.values
x = combined_df.iloc[:, 0]
y = combined_df.iloc[:, 1]
w = pd.concat([x, y], axis=1)  # Combine x and y for KMeans

# --- Elbow Method for KMeans Clustering ---
num_clusters_range = range(1, 15)
sum_of_squared_distances = []

for num_clusters in num_clusters_range:
    kmeans_temp = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)  # Ensure KMeans converges
    kmeans_temp.fit(w)
    sum_of_squared_distances.append(kmeans_temp.inertia_)

plt.figure(figsize=(16, 10))
plt.plot(num_clusters_range, sum_of_squared_distances, 'bo-')
plt.xlabel("Number of Clusters (K)", fontsize=14, fontweight='bold', fontname='serif')
plt.ylabel("Sum of Squared Distances/Inertia", fontsize=14, fontweight='bold', fontname='serif')

# Use scientific notation for y-axis ticks
formatter = ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-2, 2))
plt.gca().yaxis.set_major_formatter(formatter)

plt.tick_params(axis='y', labelsize=12)
plt.grid(True)
plt.savefig("elbow_plot_latent_space_clustering.png", dpi=150, bbox_inches='tight')


# --- KMeans Clustering ---
num_clusters = 8
kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)  # Increased n_init
kmeans.fit(w)
cluster_labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# --- Clustering Information Plot ---
plt.figure(figsize=(16, 10))
scatter = plt.scatter(x, y, c=cluster_labels, cmap='tab10', s=50)  # Color points by cluster
plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200, color='white', edgecolor='black', linewidth=2, label='Cluster Centers')

plt.xlabel("Latent Component-1", fontsize=25, fontweight='bold', fontname='serif', labelpad=15)
plt.ylabel("Latent Component-2", fontsize=25, fontweight='bold', fontname='serif', labelpad=15)
plt.xticks(fontsize=18, fontweight='bold', fontname='serif')
plt.yticks(fontsize=18, fontweight='bold', fontname='serif')

# Create a legend with cluster labels starting from 1
handles, labels = scatter.legend_elements()
offset_labels = [int(re.search(r'\d+', label).group()) + 1 for label in labels]  # Extract digits and add 1
legend = plt.legend(handles, offset_labels, title="Cluster Labels", fontsize=15, title_fontsize=20, markerscale=3, ncol=3)
legend.get_title().set_fontweight('bold')

plt.setp(legend.get_texts(), fontname='serif', fontweight='bold')
plt.setp(legend.get_title(), fontname='serif')

# Increase border width of the plot
plt.gca().spines['top'].set_linewidth(2)
plt.gca().spines['right'].set_linewidth(2)
plt.gca().spines['bottom'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)

# Adjust the length of the ticks on both axes
plt.tick_params(axis='x', which='major', length=8, width=2)
plt.tick_params(axis='y', which='major', length=8, width=2)

plt.savefig("latent_space_clustered.png", dpi=150, bbox_inches='tight')


# --- KDE on Latent Space ---
kde = gaussian_kde(latent_space.T)  # Perform KDE on the entire latent space
x_min, x_max = latent_space[:, 0].min(), latent_space[:, 0].max()
y_min, y_max = latent_space[:, 1].min(), latent_space[:, 1].max()

# Create a mesh grid for evaluation
num_points = 250  # Define number of grid points
aspect_ratio = (x_max - x_min) / (y_max - y_min)  # Calculate aspect ratio

x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, num_points), np.linspace(y_min, y_max, num_points))
grid_coords = np.vstack([x_grid.ravel(), y_grid.ravel()])  # Combine into shape (2, n_samples)

# Evaluate the KDE on the grid
z = kde(grid_coords)
z = z.reshape(x_grid.shape)  # Reshape to the grid shape for plotting

# --- Select Most Undersampled Points from Each Cluster ---
def select_undersampled_points(latent_space, kde_values, cluster_labels, num_clusters, samples_per_cluster=2):
    """Select the most undersampled points from each cluster"""
    cluster_dict = {i: [] for i in range(num_clusters)}
    
    for point, kde_val, cluster in zip(latent_space, kde_values, cluster_labels):
        cluster_dict[cluster].append((point, kde_val, cluster))  # Store point, KDE value, and cluster label
    
    # Sort points in each cluster by KDE value and select the lowest
    sampled_points_with_cluster = []
    for cluster in cluster_dict:
        points = sorted(cluster_dict[cluster], key=lambda x: x[1])[:samples_per_cluster]
        sampled_points_with_cluster.extend(points)  # Store (point, kde_val, cluster)
    
    sampled_points = np.array([item[0] for item in sampled_points_with_cluster])  # Extract just the points
    cluster_labels_for_sampled_points = np.array([item[2] for item in sampled_points_with_cluster])  # Extract cluster labels
    
    return sampled_points, cluster_labels_for_sampled_points  # Return both points and their cluster labels

# Calculate KDE values for each point in latent space
kde_values = kde(latent_space.T)

# Select the most undersampled points from each cluster
num_undersampled = 2
sampled_points, cluster_labels_for_sampled_points = select_undersampled_points(latent_space, kde_values, cluster_labels, num_clusters, samples_per_cluster=num_undersampled)

# --- Bayesian Optimization for noise_scale ---
def objective(noise_scale):
    """Objective function for Bayesian optimization"""
    noise_scale = noise_scale[0]  # Extract noise_scale from the list
    
    all_silhouette_scores = []  # List to store Silhouette scores for all clusters

    # For each undersampled point, generate new points and compute Silhouette score
    for point in sampled_points:
        # Generate new points with the given noise_scale
        generated_points = []
        for _ in range(5):
            noise = np.random.normal(0, noise_scale, size=point.shape)
            generated_points.append(point + noise)

        generated_points = np.array(generated_points)

        # Combine original and generated points for evaluation, considering only generated points this time
        eval_data = generated_points
        
        # Clustering with generated points
        kmeans_eval = KMeans(n_clusters=2, n_init=10, random_state=42) # Cluster generated points into two clusters to calculate the silhoutte score
        cluster_labels_eval = kmeans_eval.fit_predict(eval_data)  # Labels for generated points only
        
        # Scale the data before computing Silhouette Score
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(eval_data)  # Scale only the generated points

        # If there's only one unique label after clustering, return -1
        if len(np.unique(cluster_labels_eval)) < 2:
            silhouette_avg = -1
        else:
            # Calculate Silhouette Score
            silhouette_avg = silhouette_score(scaled_data, cluster_labels_eval)  # Pass scaled_data
        all_silhouette_scores.append(silhouette_avg) # Appending the silihouette score

    # Return negative mean Silhouette Score
    return -np.mean(all_silhouette_scores)

# Define the search space for noise_scale
search_space = [Real(0.01, 0.2, name='noise_scale')]

# Store noise_scale values and Silhouette scores for plotting
noise_scale_values = []
silhouette_scores = []

def callback(res):
    """Callback function to store noise_scale and Silhouette score after each iteration"""
    noise_scale_values.append(res.x[0])  # Append the noise_scale value
    silhouette_scores.append(-res.fun)  # Append the Silhouette score (negated, because gp_minimize minimizes)

# Perform Bayesian optimization, passing the callback function
res_gp = gp_minimize(objective, search_space, n_calls=50, random_state=42, callback=[callback])

# Print the optimized noise_scale
optimized_noise_scale = res_gp.x[0]
print(f"Optimized noise_scale: {optimized_noise_scale}")

# --- Generate New Points Using Optimized Noise Scale and Select Farthest ---
generated_points_all = []
farthest_points_all = []  # To store one generated point per undersampled point
cluster_labels_farthest = []  # To store cluster labels for the farthest points

for i, point in enumerate(sampled_points):
    generated_points = []
    for _ in range(50):
        noise = np.random.normal(0, optimized_noise_scale, size=point.shape)
        generated_points.append(point + noise)

    generated_points = np.array(generated_points)

    # Calculate distances from the original undersampled point
    distances = np.linalg.norm(generated_points - point, axis=1)

    # Select the farthest point
    farthest_point_index = np.argmax(distances)
    farthest_point = generated_points[farthest_point_index]
    
    generated_points_all.append(farthest_point) # Only append the farthest point
    farthest_points_all.append(farthest_point)
    cluster_labels_farthest.append(cluster_labels_for_sampled_points[i])  # Store corresponding cluster label

farthest_points_all = np.array(farthest_points_all)
cluster_labels_farthest = np.array(cluster_labels_farthest)

# --- Visualization ---
plt.figure(figsize=(10 * aspect_ratio, 10))  # Adjust figsize based on aspect ratio

cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    "", ["white", "lightblue","blue", "cyan", "lime", "yellow", "orange", "red"]
)

# Plot KDE contour
contour = plt.contourf(x_grid, y_grid, z, levels=50, cmap=cmap, alpha=0.8)  # Plot KDE as contours

# Plot cluster centers
plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200, color='black', edgecolor='black', linewidth=2, label='Cluster Centers')

# Plot undersampled points
plt.scatter(sampled_points[:, 0], sampled_points[:, 1], c='red', s=100, edgecolor='black', linewidth=1.5, label='Selected Undersampled Points')

# Plot generated points (farthest points)
plt.scatter(farthest_points_all[:, 0], farthest_points_all[:, 1], c='yellow', s=100, edgecolor='black', linewidth=1, label='Generated Points')

# --- Colorbar Formatting ---
cbar = plt.colorbar(contour, shrink=0.94)
cbar.set_label('Probability Density', fontsize=22, fontweight='bold', fontname='serif', labelpad=15)
cbar.ax.tick_params(labelsize=18, width=2, length=6)

fontprops = FontProperties(family='serif', style='normal', weight='bold', size=18)
for label in cbar.ax.get_yticklabels():
    label.set_fontproperties(fontprops)

# Plot limits and labels
plt.xlabel("Latent Component-1", fontsize=26, fontweight='bold', fontname="serif", labelpad=20)
plt.ylabel("Latent Component-2", fontsize=26, fontweight='bold', fontname="serif", labelpad=20)
# Axis styling
for spine in plt.gca().spines.values():
    spine.set_linewidth(2)
plt.xticks(fontsize=20, fontweight='bold', fontname="serif")
plt.yticks(fontsize=20, fontweight='bold', fontname="serif")
plt.tick_params(axis='both', which='major', width=2, length=6)

# --- Legend ---
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles, labels, loc='upper right', fontsize=16, frameon=True)

# Save the plot
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust tight_layout to make room for labels
plt.savefig('undersampled_regions_with_generated_points.png', dpi=300)



# Save noise_scale and its corresponding silhouette score to CSV
noise_silhouette_data = list(zip(noise_scale_values, silhouette_scores))
noise_silhouette_df = pd.DataFrame(noise_silhouette_data, columns=['Noise_Scale', 'Silhouette_Score'])

# Save to CSV
noise_silhouette_df.to_csv('noise_scale_silhouette_scores.csv', index=False)

print("Noise scale and silhouette scores saved to 'noise_scale_silhouette_scores.csv'")


# --- Plotting Silhouette Score vs Noise Scale ---
#plt.figure(figsize=(16, 10))
#plt.plot(noise_scale_values, silhouette_scores, 'bo-')
#plt.xlabel("Noise Scale Value", fontsize=14, fontweight='bold', fontname='serif')
#plt.ylabel("Silhouette Score", fontsize=14, fontweight='bold', fontname='serif')
#plt.title("Silhouette Score vs Noise Scale", fontsize=16, fontweight='bold', fontname='serif')
#plt.xticks(fontsize=12, fontweight='bold', fontname='serif')
#plt.yticks(fontsize=12, fontweight='bold', fontname='serif')
#plt.grid(True)
#plt.savefig("silhouette_score_vs_noise_scale.png", dpi=150, bbox_inches='tight')


# --- Saving data to CSV files ---
# --- Saving data to CSV files ---
# Prepare data for undersampled points
undersampled_data = []
for i, point in enumerate(sampled_points):
    # Increment cluster labels by 1
    cluster_label = cluster_labels_for_sampled_points[i] + 1
    undersampled_data.append([point[0], point[1], cluster_label])
undersampled_df = pd.DataFrame(undersampled_data, columns=['Latent_Component_1', 'Latent_Component_2', 'Cluster'])

# Prepare data for generated points
generated_data = []
for i, point in enumerate(farthest_points_all):
    # Increment cluster labels by 1
    cluster_label = cluster_labels_farthest[i] + 1
    generated_data.append([point[0], point[1], cluster_label])
generated_df = pd.DataFrame(generated_data, columns=['Latent_Component_1', 'Latent_Component_2', 'Cluster'])

# Save to CSV files
undersampled_df.to_csv('undersampled_regions.csv', index=False)
generated_df.to_csv('generated_points.csv', index=False)

print(f"Sampled points saved to 'undersampled_regions.csv'")
print(f"Generated points saved to 'generated_points.csv'")

