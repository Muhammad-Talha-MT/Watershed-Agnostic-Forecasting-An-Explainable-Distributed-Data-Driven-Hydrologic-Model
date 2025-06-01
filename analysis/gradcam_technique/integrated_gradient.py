import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import yaml
from sklearn.metrics import r2_score
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, MyMultiOutputTarget
from data_loader import HDF5Dataset

# -------------------------
# Config and cuDNN settings
# -------------------------
def load_yaml_config(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)
        
config = load_yaml_config('config/config.yaml')
# Temporarily disable cuDNN for RNNs
cudnn_enabled = cudnn.enabled
cudnn.enabled = False

# -------------------------
# Custom functions for heatmap visualization
# -------------------------
def custom_show_cam(heatmap, colormap_name='Reds', threshold=0.1):
    """
    Display a normalized heatmap (values in [0,1]) using a specified colormap.
    Values below the threshold are shown as white.
    """
    masked_heatmap = np.ma.masked_less(heatmap, threshold)
    cmap = plt.get_cmap(colormap_name).copy()
    cmap.set_bad(color='white')
    plt.figure(figsize=(8, 6))
    img = plt.imshow(masked_heatmap, cmap=cmap, vmin=threshold, vmax=heatmap.max())
    plt.title("Heatmap")
    plt.axis("off")
    plt.colorbar(img, label="Intensity")
    plt.show()
    return masked_heatmap

def custom_save_cam(heatmap, colormap_name='Reds', threshold=0.1, save_path='heatmap.png'):
    """
    Save a normalized heatmap (values in [0,1]) as an image using a specified colormap.
    Values below the threshold are shown as white.
    """
    masked_heatmap = np.ma.masked_less(heatmap, threshold)
    cmap = plt.get_cmap(colormap_name).copy()
    cmap.set_bad(color='white')
    fig, ax = plt.subplots(figsize=(8, 6))
    img = ax.imshow(masked_heatmap, cmap=cmap, vmin=threshold, vmax=heatmap.max())
    ax.set_title("Heatmap")
    ax.axis("off")
    fig.colorbar(img, ax=ax, label="Intensity")
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    return masked_heatmap

def plot_multiple_heatmaps(occurrences, magnitudes, multiplies, threshold_values, save_path):
    """
    Plot multiple heatmaps in a single figure, one for each threshold value.
    
    Parameters:
      occurrences: 3D numpy array of shape (num_thresholds, height, width)
      magnitudes: 3D numpy array of shape (num_thresholds, height, width)
      multiplies: 3D numpy array of shape (num_thresholds, height, width)
      threshold_values: List of threshold values.
      save_path: File path to save the final combined plot.
    """
    num_thresholds = len(threshold_values)
    fig, axes = plt.subplots(3, num_thresholds, figsize=(30,15))
    axes = axes.flatten()

    for i in range(num_thresholds):
        occurrence = occurrences[i]
        magnitude = magnitudes[i]
        multiple = multiplies[i]
        threshold = threshold_values[i]
        
        masked_occurrence = np.ma.masked_less(occurrence, 0.1)
        masked_magnitude = np.ma.masked_less(magnitude, 0.1)
        masked_multiple = np.ma.masked_less(multiple, 0.001)
        
        cmap_occurrence = plt.get_cmap('Reds')
        cmap_occurrence.set_bad(color='white')
        im_occurrence = axes[i].imshow(masked_occurrence, cmap=cmap_occurrence, vmin=0, vmax=1)
        axes[i].set_title(f"Occurrence\nThreshold: {threshold}", fontsize=12)
        axes[i].axis("off")
        fig.colorbar(im_occurrence, ax=axes[i], fraction=0.046, pad=0.04)
        
        cmap_magnitude = plt.get_cmap('Greens')
        cmap_magnitude.set_bad(color='white')
        im_magnitude = axes[i + num_thresholds].imshow(masked_magnitude, cmap=cmap_magnitude, vmin=0, vmax=1)
        axes[i + num_thresholds].set_title(f"Magnitude\nThreshold: {threshold}", fontsize=12)
        axes[i + num_thresholds].axis("off")
        fig.colorbar(im_magnitude, ax=axes[i + num_thresholds], fraction=0.046, pad=0.04)
        
        vmin = np.min(multiplies)
        vmax = np.max(multiplies)
        if vmax - vmin < 1e-6:
            vmax = vmin + 1e-6
        cmap_multiple = plt.get_cmap('Blues')
        cmap_multiple.set_bad(color='white')
        im_multiple = axes[i + 2 * num_thresholds].imshow(masked_multiple, cmap=cmap_multiple, vmin=vmin, vmax=vmax)
        axes[i + 2 * num_thresholds].set_title(f"Multiplied\nThreshold: {threshold}", fontsize=12)
        axes[i + 2 * num_thresholds].axis("off")
        fig.colorbar(im_multiple, ax=axes[i + 2 * num_thresholds], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=500)
    plt.show()

# -------------------------
# Model definitions
# -------------------------
class CNN_LSTM(nn.Module):
    def __init__(self):
        super(CNN_LSTM, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.batch_norm1 = nn.BatchNorm2d(16)
        self.batch_norm2 = nn.BatchNorm2d(32)
        self.batch_norm3 = nn.BatchNorm2d(64)
        self.dropout_cnn = nn.Dropout(0.1)
        # Assuming input size is 1849 x 1458
        cnn_output_height = 1849 // 8  
        cnn_output_width = 1458 // 8
        cnn_output_channels = 64
        cnn_output_size = cnn_output_channels * cnn_output_height * cnn_output_width
        # LSTM layers
        self.lstm = nn.LSTM(167040, 512, num_layers=2, batch_first=True, dropout=0.1)
        # Fully connected layers
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 61)
        nn.init.constant_(self.fc1.bias, 0.1)
        nn.init.constant_(self.fc2.bias, 0.1)
        nn.init.constant_(self.fc3.bias, 0.1)
        self.relu = nn.ReLU()

    def forward(self, ppt, tmin, tmax):
        # Concatenate inputs along a new modality dimension
        # Input shapes: [B, seq_len, H, W] each. After unsqueeze: [B, seq_len, 1, H, W]
        # After concatenation: [B, seq_len, 3, H, W]
        x = torch.cat((ppt.unsqueeze(2), tmin.unsqueeze(2), tmax.unsqueeze(2)), dim=2)
        batch_size, seq_len, _, height, width = x.shape
        cnn_features = []
        for t in range(seq_len):
            x_t = x[:, t]  # shape: [B, 3, H, W]
            x_t = self.pool(self.batch_norm1(F.relu(self.conv1(x_t))))
            x_t = self.pool(self.batch_norm2(F.relu(self.conv2(x_t))))
            x_t = self.pool(self.batch_norm3(F.relu(self.conv3(x_t))))
            x_t = x_t.view(batch_size, -1)  # flatten
            cnn_features.append(x_t)
        x = torch.stack(cnn_features, dim=1)  # [B, seq_len, cnn_output_size]
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # using last time step
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class GradCAMWrapper(nn.Module):
    def __init__(self, model):
        super(GradCAMWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        # x shape: [B, 3, seq_len, H, W]
        ppt = x[0, :, :, :]   # [B, seq_len, H, W]
        tmin = x[1, :, :, :]   # [B, seq_len, H, W]
        tmax = x[2, :, :, :]   # [B, seq_len, H, W]
        ppt = ppt.unsqueeze(0)
        tmin = tmin.unsqueeze(0)
        tmax = tmax.unsqueeze(0)
        return self.model(ppt, tmin, tmax)

# -------------------------
# Load Model and Dataset
# -------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = CNN_LSTM().to(device)
checkpoint_path = '/home/talhamuh/water-research/CNN-LSMT/src/cnn_lstm_project/checkpoint/finetune_cnn_lstm_global_temporal_extrapolation_61_10y'
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['state_dict'])
print("Pretrained checkpoint loaded successfully!")

variables_to_load = ['ppt', 'tmin', 'tmax']
dataset = HDF5Dataset(config['h5_file'], variables_to_load, config['labels_path'], 2000, 2009)
dataset_size = len(dataset)
num_train = int(0.8 * len(dataset))
num_val = int(0.1 * len(dataset))
num_test = len(dataset) - num_train - num_val
train_indices = range(0, num_train)
val_indices = range(num_train, num_train + num_val)
test_indices = range(num_train + num_val, dataset_size)
train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)
test_dataset = Subset(dataset, test_indices)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=32)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=32)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=32)
print("data has been loaded ...")
wrapped_model = GradCAMWrapper(model).to(device)

# -------------------------
# Run Inference with Grad-CAM and Process Heatmaps
# -------------------------
def run_inference_with_gradcam(model, wrapped_model, test_loader, device, save_dir, cam_target_layer, target_category, threshold=0.1):
    """
    Run inference + Grad-CAM on each batch in test_loader.
    """
    model.eval()
    wrapped_model.eval()
    cam = GradCAM(model=wrapped_model, target_layers=[cam_target_layer])
    all_outputs = []
    all_labels = []
    all_heatmaps = []
    for batch_idx, batch in enumerate(test_loader):
        ppt = batch['ppt'].to(device)       # [B, seq_len, H, W]
        tmin = batch['tmin'].to(device)
        tmax = batch['tmax'].to(device)
        labels = batch['label'].to(device)    # [B, 61] assumed
        outputs = model(ppt, tmin, tmax)
        outputs_cpu = outputs.detach().cpu().numpy()
        labels_cpu = labels.detach().cpu().numpy()
        all_outputs.append(outputs_cpu)
        all_labels.append(labels_cpu)
        # Combine modalities into shape [B, 3, seq_len, H, W]
        day_input = torch.stack([ppt, tmin, tmax], dim=1)
        for i in range(day_input.shape[0]):
            sample_input = day_input[i]  # shape [3, seq_len, H, W]
            # Run Grad-CAM for the chosen target_category
            grayscale_cam = cam(input_tensor=sample_input, targets=[MyMultiOutputTarget(target_category)])
            cam_image = grayscale_cam[0, :]  # expected shape [H, W] or [seq_len, H, W]
            all_heatmaps.append(cam_image)
            del sample_input, grayscale_cam, cam_image
            torch.cuda.empty_cache()
        del day_input, ppt, tmin, tmax, labels
        torch.cuda.empty_cache()
    all_outputs = np.concatenate(all_outputs, axis=0)  # shape [num_samples, 61]
    all_labels = np.concatenate(all_labels, axis=0)    # shape [num_samples, 61]
    overall_r2 = r2_score(all_labels[:, 0], all_outputs[:, 0])
    print("Overall R2:", overall_r2)
    return all_heatmaps, all_outputs, all_labels

# -------------------------
# For each target (gauge) run Grad-CAM inference and generate threshold heatmaps.
# -------------------------
target_indices = range(61)  # gauges 0 to 60
for idx in target_indices:
    all_cams = []
    all_heatmaps, all_outputs, all_labels = run_inference_with_gradcam(
        model, 
        wrapped_model,
        test_loader, 
        device, 
        save_dir='/home/talhamuh/water-research/CNN-LSMT/src/cnn_lstm_project/results/02202025/gradcam/',
        cam_target_layer = model.conv3,
        target_category = idx,
        threshold = 0.1
    )
    stacked_cams = np.stack(all_heatmaps, axis=0)  # shape [num_samples, H, W]
    all_cams.append(stacked_cams)
    all_cams = np.array(all_cams)
    # Flatten across target indices if needed:
    all_cams = np.concatenate(all_cams, axis=0)
    
    threshold_vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    occurrences = []
    magnitudes = []
    multiplies = []
    for t in threshold_vals:
        total_days = all_cams.shape[0]
        occurrence_sum = np.sum(all_cams > t, axis=0)  # shape [H, W]
        combined_mean_cam = np.mean(all_cams * (all_cams > t), axis=0)
        normalized_occurrence = occurrence_sum / total_days
        multiplied_result = combined_mean_cam * normalized_occurrence
        occurrences.append(normalized_occurrence)
        magnitudes.append(combined_mean_cam)
        multiplies.append(multiplied_result)
    
    save_path = f"/home/talhamuh/water-research/CNN-LSMT/src/cnn_lstm_project/results/02202025/gradcam-2/combined_threshold_heatmap_{idx}.png"
    plot_multiple_heatmaps(occurrences, magnitudes, multiplies, threshold_vals, save_path)
    print("Normalized occurrence heatmap saved successfully for target index", idx)