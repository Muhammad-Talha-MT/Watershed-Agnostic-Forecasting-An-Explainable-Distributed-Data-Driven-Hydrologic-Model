import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.ndimage import gaussian_filter
from sklearn.metrics import r2_score
import yaml
import os
import matplotlib.pyplot as plt
from data_loader import HDF5Dataset
from model import CNN_LSTM
from scipy import stats

def load_yaml_config(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)
config = load_yaml_config('config/config.yaml')

def apply_gaussian_perturbation(data, location, std_dev, amplitude=1, flip_sign=False):
    """
    Apply a Gaussian perturbation to the input data.
    Parameters:
    - data: 4D array (batch, time, latitude, longitude) of weather data.
    - location: Tuple (lat_idx, lon_idx) indicating the center of the Gaussian perturbation.
    - std_dev: Standard deviation of the Gaussian kernel.
    - amplitude: Amplitude of the Gaussian perturbation.
    - flip_sign: Boolean, if True the perturbation is subtracted, otherwise added.
    Returns:
    - perturbed_data: 4D array of perturbed weather data.
    """
    batch, time, lat, lon = data.shape
    x = np.arange(lon)
    y = np.arange(lat)
    x, y = np.meshgrid(x, y)
    gaussian = np.exp(-((x-location[1])**2 + (y-location[0])**2) / (2 * std_dev**2))
    gaussian = amplitude * gaussian / gaussian.max()  # Normalize the Gaussian
    perturbed_data = np.copy(data)
    for b in range(batch):
        for t in range(time):
            if flip_sign:
                perturbed_data[b, t] -= gaussian
            else:
                perturbed_data[b, t] += gaussian
    return perturbed_data

def run_inference(model, data_loader, device):
    model.eval()
    all_outputs = []
    for batch in data_loader:
        ppt = batch['ppt'].to(device)
        tmin = batch['tmin'].to(device)
        tmax = batch['tmax'].to(device)
        outputs = model(ppt, tmin, tmax)
        all_outputs.append(outputs.detach().cpu().numpy())
    return np.concatenate(all_outputs, axis=0)

def calculate_sensitivity(original_outputs, perturbed_outputs):
    """
    Calculate the sensitivity as mean absolute difference between original and perturbed outputs.
    
    Parameters:
    - original_outputs: numpy array of shape (num_samples, 61)
    - perturbed_outputs: numpy array of shape (num_samples, 61)
    
    Returns:
    - sensitivity: numpy array of shape (61,)
    """
    return np.mean(np.abs(perturbed_outputs - original_outputs), axis=0)

def generate_sensitivity_maps(sensitivities, save_path, ks_distances=None, area_fractions=None):
    num_locations = sensitivities.shape[0]  # Assume shape is (61, lat, lon)
    for i in range(num_locations):
        plt.imshow(sensitivities[i], cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.title(f"Sensitivity Map for Streamflow Location {i+1}")
        plt.savefig(f"{save_path}sensitivity_output_{i+1}.png")
        plt.close()

def calculate_sensitivity_metrics(sensitivities, watershed_mask):
    """
    Calculate Kolmogorov-Smirnov distance and area fraction metrics for sensitivity maps.
    
    Parameters:
    - sensitivities: numpy array of shape (61, lat, lon) containing sensitivity values
    - watershed_mask: boolean array of shape (lat, lon) indicating watershed locations
    
    Returns:
    - ks_distances: array of K-S distances for each gauge
    - area_fractions: array of area fractions above half max sensitivity
    """
    ks_distances = []
    area_fractions = []
    
    for gauge_idx in range(sensitivities.shape[0]):
        # Get sensitivities for current gauge
        gauge_sens = sensitivities[gauge_idx]
        
        # Calculate K-S distance between distributions inside and outside watershed
        inside_dist = gauge_sens[watershed_mask].flatten()
        outside_dist = gauge_sens[~watershed_mask].flatten()
        ks_stat = stats.ks_2samp(inside_dist, outside_dist).statistic
        ks_distances.append(ks_stat)
        
        # Calculate area fraction above half max sensitivity
        half_max = gauge_sens.max() / 2
        area_frac = np.mean(gauge_sens > half_max)
        area_fractions.append(area_frac)
        
    return np.array(ks_distances), np.array(area_fractions)

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CNN_LSTM().to(device)
    checkpoint_path = "/home/talhamuh/water-research/CNN-LSMT/src/cnn_lstm_project/checkpoint/finetune_cnn_lstm_global_temporal_extrapolation_61_10y"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    dataset = HDF5Dataset(config['h5_file'], ['ppt', 'tmin', 'tmax'], config['labels_path'], 2009, 2009)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    # Debugging: print the shape of 'ppt' tensor
    sample = next(iter(test_loader))
    print("Shape of 'ppt' tensor:", sample['ppt'].shape)  # Check the actual shape

    # Assuming the shape is [batch_size, seq_length, lat, lon]
    # Adjust this based on your actual data structure:
    batch_size, seq_length, lat, lon = sample['ppt'].shape
    sensitivities = np.zeros((61, lat, lon))  # Initialize sensitivity array

    original_outputs = run_inference(model, test_loader, device)

    # Grid of perturbation locations
    x_positions = np.linspace(0, lon-1, 20, dtype=int)
    y_positions = np.linspace(0, lat-1, 20, dtype=int)
    
    # Iterate over perturbation locations
    for x in x_positions:
        for y in y_positions:
            perturbed_outputs = []
            # for batch in test_loader:
            ppt = sample['ppt'].to(device)
            perturbed_ppt = apply_gaussian_perturbation(
                ppt.cpu().numpy(), 
                location=(y, x),  # (lat, lon) coordinates
                std_dev=10
            )
            perturbed_ppt = torch.tensor(perturbed_ppt, dtype=torch.float32).to(device)
            output = model(perturbed_ppt, sample['tmin'].to(device), sample['tmax'].to(device))
            perturbed_outputs.append(output.detach().cpu().numpy())
        
        perturbed_outputs = np.concatenate(perturbed_outputs, axis=0)
        sensitivity = calculate_sensitivity(original_outputs, perturbed_outputs)
        
        # Assign sensitivity values to the corresponding location in the map
        for i in range(61):  # For each gauge
            sensitivities[i, y, x] = sensitivity[i]

    # Create output directory if it doesn't exist
    save_path = '/home/talhamuh/water-research/CNN-LSMT/src/cnn_lstm_project/results/02202025/sensitivity_maps/'
    os.makedirs(save_path, exist_ok=True)
    
    # Generate and save sensitivity maps
    generate_sensitivity_maps(
        sensitivities, 
        save_path=save_path,
        ks_distances=None,
        area_fractions=None
    )
    
    print("Sensitivity maps generated successfully!")

    # Load or create watershed masks (placeholder - needs actual watershed data)
    watershed_masks = np.ones((61, lat, lon), dtype=bool)  # Replace with actual watershed masks
    
    # Calculate metrics
    ks_distances, area_fractions = calculate_sensitivity_metrics(sensitivities, watershed_masks)
    
    generate_sensitivity_maps(
        sensitivities, 
        save_path='/home/talhamuh/water-research/CNN-LSMT/src/cnn_lstm_project/results/02202025/sensitivity_maps',
        ks_distances=ks_distances,
        area_fractions=area_fractions
    )

if __name__ == "__main__":
    main()

