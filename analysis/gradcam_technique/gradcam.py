import matplotlib.pyplot as plt
from pytorch_grad_cam.utils.image import show_cam_on_image  # you can omit if using custom overlay
import cv2
import numpy as np
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, MyMultiOutputTarget
import torch
from pytorch_grad_cam import GradCAM
import sys
import numpy
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.backends.cudnn as cudnn
import yaml
import torch.nn.functional as F
import os
from sklearn.metrics import r2_score

def load_yaml_config(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)
config = load_yaml_config('config/config.yaml')
# Temporarily disable cuDNN for RNNs
cudnn_enabled = cudnn.enabled  # save current setting
cudnn.enabled = False
def custom_show_cam(heatmap, colormap_name='Reds', threshold=0.1):
    """
    Display a normalized heatmap (values in [0,1]) using a specified colormap.
    Values below the threshold are shown as white.
    
    Parameters:
      heatmap: 2D numpy array with normalized values in [0,1].
      colormap_name: Name of the matplotlib colormap to use.
      threshold: Values below this threshold are displayed as white.
    """
    # Create a masked array that masks values below the threshold
    masked_heatmap = np.ma.masked_less(heatmap, 0)
    # masked_heatmap = heatmap
    
    # Get a copy of the desired colormap and set the color for masked values to white
    cmap = plt.get_cmap(colormap_name).copy()
    cmap.set_bad(color='white')
    
    # Display the heatmap
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
    
    Parameters:
      heatmap: 2D numpy array with normalized values in [0,1].
      colormap_name: Name of the matplotlib colormap to use.
      threshold: Values below this threshold are displayed as white.
      save_path: File path where the plot will be saved.
    """
    # Create a masked array that masks values below the threshold.
    masked_heatmap = np.ma.masked_less(heatmap, threshold)
    
    # Get a copy of the desired colormap and set the color for masked values to white.
    cmap = plt.get_cmap(colormap_name).copy()
    cmap.set_bad(color='white')
    
    # Create a new figure and plot the heatmap.
    fig, ax = plt.subplots(figsize=(8, 6))
    img = ax.imshow(masked_heatmap, cmap=cmap, vmin=threshold, vmax=heatmap.max())
    ax.set_title("Heatmap")
    ax.axis("off")
    fig.colorbar(img, ax=ax, label="Intensity")
    
    # Save the plot to the specified file.
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    
    return masked_heatmap

def plot_multiple_heatmaps(occurrences, magnitudes, multiplies, threshold_values, save_path):
    """
    Plot multiple heatmaps in a single figure, one for each threshold value.

    Parameters:
        occurrences: 3D numpy array of shape (num_thresholds, height, width) containing normalized occurrence heatmaps.
        magnitudes: 3D numpy array of shape (num_thresholds, height, width) containing magnitude heatmaps.
        threshold_values: List of threshold values.
        save_path: Path to save the final combined plot.
    """
    num_thresholds = len(threshold_values)  # Number of thresholds

    fig, axes = plt.subplots(3, num_thresholds, figsize=(30, 15))  # 2 rows for occurrences and magnitudes
    axes = axes.flatten()  # Flatten for easy access

    for i in range(num_thresholds):
        occurrence = occurrences[i]
        magnitude = magnitudes[i]
        threshold = threshold_values[i]
        multiple = multiplies[i]

        # Mask values below a threshold for better visualization
        masked_occurrence = np.ma.masked_less(occurrence, 0.1)
        masked_magnitude = np.ma.masked_less(magnitude, 0.1)
        masked_multiple = np.ma.masked_less(multiple, 0.001)

        # Plot occurrence heatmap
        cmap_occurrence = plt.get_cmap('Reds')
        cmap_occurrence.set_bad(color='white')  # Set background to white for masked values
        im_occurrence = axes[i].imshow(masked_occurrence, cmap=cmap_occurrence, vmin=0, vmax=1)
        axes[i].set_title(f"Occurrence\nThreshold: {threshold}", fontsize=12)
        axes[i].axis("off")
        fig.colorbar(im_occurrence, ax=axes[i], fraction=0.046, pad=0.04)

        # Plot magnitude heatmap
        cmap_magnitude = plt.get_cmap('Greens')
        cmap_magnitude.set_bad(color='white')  # Set background to white for masked values
        im_magnitude = axes[i + num_thresholds].imshow(masked_magnitude, cmap=cmap_magnitude, vmin=0, vmax=1)
        axes[i + num_thresholds].set_title(f"Magnitude\nThreshold: {threshold}", fontsize=12)
        axes[i + num_thresholds].axis("off")
        fig.colorbar(im_magnitude, ax=axes[i + num_thresholds], fraction=0.046, pad=0.04)
        
        
        vmin = np.min(multiplies)  # Adjust this if necessary
        vmax = np.max(multiplies)  # Adjust this if necessary
        print(vmin, vmax)
        # Avoid vmax being too close to vmin
        if vmax - vmin < 1e-6:
            vmax = vmin + 1e-6  # Ensure a valid range
        # Plot multiplies heatmap
        cmap_multiple = plt.get_cmap('Blues')
        cmap_multiple.set_bad(color='white')  # Set background to white for masked values
        im_magnitude = axes[i + 2 * num_thresholds].imshow(masked_multiple, cmap=cmap_multiple, vmin=vmin, vmax=vmax)
        axes[i + 2 * num_thresholds].set_title(f"Multiplied Values\nThreshold: {threshold}", fontsize=12)
        axes[i + 2 * num_thresholds].axis("off")
        fig.colorbar(im_magnitude, ax=axes[i + 2 * num_thresholds], fraction=0.046, pad=0.04)

    # Adjust layout and save final plot
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=500)
    plt.show()
    
    
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
        
        # Compute CNN output dimensions based on your input size.
        # Here, these values are placeholders – adjust them to match your data.
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
        
        # Initialize biases
        nn.init.constant_(self.fc1.bias, 0.1)
        nn.init.constant_(self.fc2.bias, 0.1)
        nn.init.constant_(self.fc3.bias, 0.1)
        
        self.relu = nn.ReLU()

    def forward(self, ppt, tmin, tmax):
        # Concatenate inputs along the channel dimension
        x = torch.cat((ppt.unsqueeze(2), tmin.unsqueeze(2), tmax.unsqueeze(2)), dim=2)
        batch_size, seq_len, _, height, width = x.shape
        
        # Process each timestep independently through the CNN
        cnn_features = []
        for t in range(seq_len):
            x_t = x[:, t]  # [batch_size, 3, height, width]
            x_t = self.pool(self.batch_norm1(F.relu(self.conv1(x_t))))
            x_t = self.pool(self.batch_norm2(F.relu(self.conv2(x_t))))
            x_t = self.pool(self.batch_norm3(F.relu(self.conv3(x_t))))
            x_t = x_t.view(batch_size, -1)  # Flatten for LSTM input
            cnn_features.append(x_t)
        
        # Stack features along the time dimension
        x = torch.stack(cnn_features, dim=1)  # [batch_size, seq_len, cnn_output_size]
        
        # Process with LSTM
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Use last time step
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

class GradCAMWrapper(nn.Module):
    def __init__(self, model):
        super(GradCAMWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        # x shape: [B, 3, 5, H, W]
        # Extract each modality along the channel dimension.
        ppt = x[0, :, :, :]   # shape: [B, 5, H, W]
        tmin = x[1, :, :, :]  # shape: [B, 5, H, W]
        tmax = x[2, :, :, :]  # shape: [B, 5, H, W]
        # If your model expects a channel dimension for each modality, add it.
        # This converts the shape to [B, 5, 1, H, W].
        ppt = ppt.unsqueeze(0)
        tmin = tmin.unsqueeze(0)
        tmax = tmax.unsqueeze(0)
        return self.model(ppt, tmin, tmax)


# Create model instance and move to device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = CNN_LSTM().to(device)
# Define the path to your checkpoint file
checkpoint_path = '/home/talhamuh/water-research/CNN-LSMT/src/cnn_lstm_project/checkpoint/finetune_cnn_lstm_global_temporal_extrapolation_61_10y'

# Load the checkpoint using torch.load
checkpoint = torch.load(checkpoint_path, map_location=device)

# Load the state dictionary into your model
model.load_state_dict(checkpoint['state_dict'])

# Set the model to evaluation mode
# model.eval()

print("Pretrained checkpoint loaded successfully!")
    
from data_loader import HDF5Dataset
variables_to_load = ['ppt', 'tmin', 'tmax']
dataset = HDF5Dataset(config['h5_file'], variables_to_load, config['labels_path'], 2000, 2009)  
dataset_size = len(dataset)    
num_train = int(0.8 * len(dataset))
num_val = int(0.1 * len(dataset))
num_test = len(dataset) - num_train - num_val

# Indices:
train_indices = range(0, num_train)
val_indices   = range(num_train, num_train + num_val)
test_indices  = range(num_train + num_val, dataset_size)
# Randomly split the dataset
# train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [num_train, num_val, num_test])
# Create subsets
train_dataset = Subset(dataset, train_indices)
val_dataset   = Subset(dataset, val_indices)
test_dataset  = Subset(dataset, test_indices)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=32)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=32)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=32)

# test_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=32)
print("data has been loaded ...")
# Wrap the model
wrapped_model = GradCAMWrapper(model).to(device)
# wrapped_model.eval()

# # Choose the target layer for Grad-CAM (using conv3 as an example)
# target_layer = model.conv3

# # Initialize the GradCAM object; use CUDA if available
# cam = GradCAM(model=wrapped_model, target_layers=[target_layer])


def run_inference_with_gradcam(
    model, 
    wrapped_model,
    test_loader, 
    device, 
    save_dir,
    cam_target_layer,
    target_category,
    threshold=0.1
):
    """
    Run inference + Grad-CAM on each batch in test_loader.
    """
    model.eval()  # put model in eval mode
    wrapped_model.eval()
    # Initialize GradCAM object
    cam = GradCAM(model=wrapped_model, target_layers=[cam_target_layer])
    all_outputs = []
    all_labels = []
    all_heatmaps = []
    # -----------------------------
    # Main loop over test data
    # -----------------------------
    for batch_idx, batch in enumerate(test_loader):
        # 1) Move data to device
        ppt  = batch['ppt'].to(device)   # [B, seq_len, H, W]
        tmin = batch['tmin'].to(device)  # [B, seq_len, H, W]
        tmax = batch['tmax'].to(device)  # [B, seq_len, H, W]
        labels = batch['label'].to(device)  # [B, some_output_dim]
        # exit()
        # 2) Model forward pass (inference on the full batch)
        #    This is your typical inference call
        outputs = model(ppt, tmin, tmax)
        
        # Collect outputs and labels (for metrics later, if needed)
        outputs_cpu = outputs.detach().cpu().numpy()  
        labels_cpu = labels.detach().cpu().numpy()        
        all_outputs.append(outputs_cpu)
        all_labels.append(labels_cpu)
        
        # print(r2_score(all_labels, all_outputs))
        # -----------------------------
        # Grad-CAM per sample
        # -----------------------------
        # Combine modalities into shape [B, 3, seq_len, H, W]
        day_input = torch.stack([ppt, tmin, tmax], dim=1)  # day_input: [B, 3, seq_len, H, W]
        
        # Loop over each sample in the batch to generate Grad-CAM
        for i in range(day_input.shape[0]):
            sample_input = day_input[i]  # shape [1, 3, seq_len, H, W]

            # If your GradCAM target is an output index, define or pass it here
            # For multi-output, you might implement custom logic in the target:
            # e.g., MyMultiOutputTarget(target_category) 
            # or just ClassifierOutputTarget(0) 
            
            
            # 3) Run Grad-CAM on the sample
            #    The shape must match the model's expected input shape
            grayscale_cam = cam(
                input_tensor=sample_input, 
                targets=[MyMultiOutputTarget(target_category)]
            )

            # 4) Visualize or save the Grad-CAM
            cam_image = grayscale_cam[0, :]  # shape: [seq_len, H, W] or [H, W], depending on your model
            # # Adjust your custom function calls accordingly:
            # masked_heatmap = custom_show_cam(cam_image, colormap_name='Reds', threshold=threshold)
            all_heatmaps.append(cam_image)
            # # (Optional) Save the heatmap
            # save_path = os.path.join(save_dir, f"gradcam_{batch_idx}_{i}.png")
            # custom_save_cam(cam_image, colormap_name='Reds', threshold=threshold, save_path=save_path)

            # Clean up
            del sample_input, grayscale_cam, cam_image
            torch.cuda.empty_cache()

        # Clean up after each batch
        del day_input, ppt, tmin, tmax, labels
        torch.cuda.empty_cache()

    # -----------------------------------
    # After iterating over all batches
    # you can compute metrics, etc.
    # -----------------------------------
    # all_outputs = torch.cat(all_outputs, dim=0)
    # all_labels  = torch.cat(all_labels,  dim=0)

    # # Example: computing MSE or other metrics on all_outputs vs. all_labels
    # # ...
    # # mse = torch.mean((all_outputs - all_labels) ** 2).item()
    # # print("MSE:", mse)
    # After finishing all batches, concatenate the arrays
    all_outputs = np.concatenate(all_outputs, axis=0)  # shape [num_samples, 61]
    all_labels  = np.concatenate(all_labels, axis=0)   # shape [num_samples, 61]
    
    # Now compute overall R^2 for one dimension
    from sklearn.metrics import r2_score
    target_category = 0  # whichever dimension you care about
    overall_r2 = r2_score(all_labels[:, target_category], all_outputs[:, target_category])
    print("Overall R2:", overall_r2)
    
    return all_heatmaps, all_outputs, all_labels

def run_inference_with_gradcam_per_sequence(
    model, 
    wrapped_model,
    test_loader, 
    device, 
    save_dir,
    location_idx,
    cam_target_layer,
    threshold=0.1
):
    """
    Run inference + Grad-CAM on each sequence separately in test_loader.
    Generates and saves individual heatmaps for each sequence.
    """
    model.eval()  # put model in eval mode
    wrapped_model.eval()
    
    # Initialize GradCAM object
    cam = GradCAM(model=wrapped_model, target_layers=[cam_target_layer])
    
    # Iterate over each batch (assuming batch_size=1 for individual sequences)
    for batch_idx, batch in enumerate(test_loader):
        ppt  = batch['ppt'].to(device)   # [B, seq_len, H, W]
        tmin = batch['tmin'].to(device)  # [B, seq_len, H, W]
        tmax = batch['tmax'].to(device)  # [B, seq_len, H, W]
        labels = batch['label'].to(device)  # [B, some_output_dim]
        outputs = model(ppt, tmin, tmax)  # Forward pass
        
        # Stack modalities into shape [B, 3, seq_len, H, W]
        day_input = torch.stack([ppt, tmin, tmax], dim=1)  # [B, 3, seq_len, H, W]

        # Loop over each sample in the batch (assuming batch_size=1)
        for i in range(day_input.shape[0]):
            sample_input = day_input[i]  # Shape: [3, seq_len, H, W]
            day_mask = sample_input[1, 1, :, :] <= 1e-6
            day_mask = day_mask.cpu().numpy()
            # Generate Grad-CAM heatmap
            grayscale_cam = cam(
                input_tensor=sample_input, 
                targets=[MyMultiOutputTarget(location_idx)]  # Assuming target category 0; modify if needed
            )

            # Extract Grad-CAM heatmap
            cam_image = grayscale_cam[0, :]  # Shape: [H, W]
            masked_cam = np.where(day_mask, 0, cam_image)
            # Save the heatmap for this sample
            save_path = os.path.join(save_dir, f"gradcam_sequence_{batch_idx}.png")
            custom_save_cam(masked_cam, colormap_name='Reds', threshold=threshold, save_path=save_path)
            print(f"Saved heatmap for sequence {batch_idx} at {save_path}")

        # Clean up
        del ppt, tmin, tmax, labels, day_input, sample_input, grayscale_cam, cam_image
        torch.cuda.empty_cache()

# target_indices = range(61)
# for idx in target_indices:
#     all_cams = []
#     all_heatmaps, all_outputs, all_labels = run_inference_with_gradcam(
#                                         model, 
#                                         wrapped_model,
#                                         test_loader, 
#                                         device, 
#                                         save_dir='/home/talhamuh/water-research/CNN-LSMT/src/cnn_lstm_project/results/02202025/gradcam/',
#                                         cam_target_layer = model.conv3,
#                                         target_category=idx,
#                                         threshold=0.1
#                                     )


#     stacked_cams = np.stack(all_heatmaps, axis=0)  # shape [num_samples, H', W']
#     all_cams.append(stacked_cams)

#     all_cams = np.array(all_cams)   
#     # Convert all heatmaps into a single array of shape [num_samples * num_targets, H', W']
#     all_cams = np.concatenate(all_cams, axis=0)  # Flatten across target indices

#     threshold = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

#     # threshold = 0.5

#     occurrences = []
#     magnitudes = []
#     mulitplies = []
#     for t in threshold:

#         total_days = all_cams.shape[0]  # Number of days (732)
#         # print(all_cams[0].shape)
#         # Count occurrences where value is greater than the threshold
#         occurrence_sum = np.sum(all_cams > t, axis=0)  # Shape (1849, 1458)
#         # Compute the combined mean heatmap across all targets and samples
#         # combined_mean_cam = np.mean(all_cams > t, axis=0)  # shape [H', W']
#         combined_mean_cam = np.mean(all_cams * (all_cams > t), axis=0)


#         # Normalize by total number of days
#         normalized_occurrence = occurrence_sum / total_days  # Shape (1849, 1458)
#         # print(combined_mean_cam)
#         # print(normalized_occurrence)
#         multiplied_result = combined_mean_cam * normalized_occurrence  # Shape (H', W')


#         # exit()
#         occurrences.append(normalized_occurrence)
#         magnitudes.append(combined_mean_cam)
#         mulitplies.append(multiplied_result)
        

#     # Save the final normalized heatmap
#     # custom_save_cam(normalized_occurrence, 'Reds', 0.1, f"/home/talhamuh/water-research/CNN-LSMT/src/cnn_lstm_project/results/02202025/gradcam/occurrence_cam_s{target_indices[0]}_t{threshold}.png")

#     # Save path for the combined plot
#     save_path = f"/home/talhamuh/water-research/CNN-LSMT/src/cnn_lstm_project/results/02202025/gradcam-1/combined_threshold_heatmap_{idx}.png"

#     # Generate and save the plot
#     plot_multiple_heatmaps(occurrences, magnitudes, mulitplies, threshold, save_path)


#     print("Normalized occurrence heatmap saved successfully!")

for location_idx in range(61):
    # Define save directory
    save_dir = f"/home/talhamuh/water-research/CNN-LSMT/src/cnn_lstm_project/results/02202025/location{location_idx}/"
    os.makedirs(save_dir, exist_ok=True)

    # Run function
    run_inference_with_gradcam_per_sequence(
        model, 
        wrapped_model,
        test_loader, 
        device, 
        save_dir,
        location_idx,
        cam_target_layer=model.conv3,
        threshold=0.1
    )