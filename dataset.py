# dataset.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
from torch.utils.data import Dataset
import os


class ClimateDataset(Dataset):
    def __init__(self, ppt, tmin, tmax, labels):
        self.ppt = torch.tensor(ppt, dtype=torch.float32)
        self.tmin = torch.tensor(tmin, dtype=torch.float32)
        self.tmax = torch.tensor(tmax, dtype=torch.float32)
        # self.labels = torch.tensor(labels.iloc[:, [1, 2, 3, 4]].apply(pd.to_numeric, errors='coerce').values, dtype=torch.float32)
        self.labels = torch.tensor(labels.apply(pd.to_numeric, errors='coerce').values, dtype=torch.float32)
        # Handle NaNs
        self.ppt = torch.nan_to_num(self.ppt, nan=0.0)
        self.tmin = torch.nan_to_num(self.tmin, nan=0.0)
        self.tmax = torch.nan_to_num(self.tmax, nan=0.0)
        self.labels = torch.nan_to_num(self.labels, nan=0.0)
        # Apply normalization globally
        # Mask zero values and normalize ppt, tmin, tmax using dim=(0, 1, 2)
        self.ppt = self._normalize_globally(self.ppt)
        self.tmin = self._normalize_globally(self.tmin)
        self.tmax = self._normalize_globally(self.tmax)
        
        # Perform min-max normalization on each frame
        # self.ppt = self._normalize_ppt_frames(self.ppt)
        # self.tmin = self._normalize_temp_frames(self.tmin)
        # self.tmax = self._normalize_temp_frames(self.tmax)
        
        # self.ppt = self._normalize_ppt_frames_pixel(self.ppt)
        # self.tmin = self._normalize_temp_frames_pixel(self.tmin)
        # self.tmax = self._normalize_temp_frames_pixel(self.tmax)
        # Perform min-max normalization on each label separately
        
        self.labels = self._normalize_labels(self.labels)
        
    
    def _normalize_globally(self, tensor):
        # Calculate the overall minimum and maximum values from the entire tensor
        overall_min = torch.min(tensor[tensor != 0])  # Minimum of non-zero values
        overall_max = torch.max(tensor[tensor != 0])  # Maximum of non-zero values
        
        # Create a copy of the tensor to hold the normalized values
        normalized_tensor = torch.empty_like(tensor)
        
        # Define a small value to assign when all values are zero
        small_value = 1e-6  # You can adjust this value as needed
        
        # Normalize using the overall min/max
        for i in range(tensor.shape[0]):  # Loop over each frame
            if torch.all(tensor[i] == 0):  # Check if all values are zero
                normalized_tensor[i] = small_value  # Assign a small value
            else:
                normalized_tensor[i] = tensor[i]  # Initialize with original tensor
                # Normalize only non-zero values
                mask = tensor[i] != 0
                normalized_tensor[i][mask] = (tensor[i][mask] - overall_min) / (overall_max - overall_min)

        return normalized_tensor


    
        
    def log_normalize(self):
        # Log normalization for ppt
        self.ppt = torch.log1p(self.ppt)  # log1p(x) = log(1 + x)

        # Log normalization for tmin and tmax with handling negative values
        self.tmin = torch.sign(self.tmin) * torch.log1p(torch.abs(self.tmin))
        self.tmax = torch.sign(self.tmax) * torch.log1p(torch.abs(self.tmax))
    
    def _normalize_ppt_frames(self, tensor):
        normalized_tensor = torch.empty_like(tensor)
        small_value = 1e-10  # Define a small negligible value
        
        for i in range(tensor.shape[0]):  # Loop over each frame
            if torch.all(tensor[i] == 0):  # Check if all values are zero
                normalized_tensor[i] = torch.full_like(tensor[i], small_value)
            else:
                min_val = torch.min(tensor[i])
                max_val = torch.max(tensor[i])
                normalized_tensor[i] = (tensor[i] - min_val) / (max_val - min_val)
        
        return normalized_tensor

    def _normalize_ppt_frames_pixel(self, tensor):
        # tensor shape is [time, height, width], so we normalize each pixel across the time dimension
        small_value = 1e-10  # Define a small negligible value
        
        # Initialize an empty tensor for normalized values
        normalized_tensor = torch.empty_like(tensor)
        
        # Iterate over each pixel (height, width)
        for i in range(tensor.shape[1]):  # Height
            for j in range(tensor.shape[2]):  # Width
                pixel_values = tensor[:, i, j]  # Get the time series for the current pixel
                
                if torch.all(pixel_values == 0):  # If all values are zero, assign the small value
                    normalized_tensor[:, i, j] = small_value
                else:
                    min_val = torch.min(pixel_values)
                    max_val = torch.max(pixel_values)
                    
                    # Normalize the time series for the current pixel
                    normalized_tensor[:, i, j] = (pixel_values - min_val) / (max_val - min_val + small_value)  # Adding small value to avoid division by zero
        
        return normalized_tensor
    
    def _normalize_temp_frames_pixel(self, tensor):
        # tensor shape is [time, height, width], so we normalize each pixel across the time dimension
        small_value = 1e-10  # Define a small negligible value to avoid division by zero
        
        # Initialize an empty tensor for normalized values
        normalized_tensor = torch.empty_like(tensor)
        
        # Iterate over each pixel (height, width)
        for i in range(tensor.shape[1]):  # Height
            for j in range(tensor.shape[2]):  # Width
                pixel_values = tensor[:, i, j]  # Get the time series for the current pixel
                
                if torch.all(pixel_values == 0):  # If all values are zero, assign a small value
                    normalized_tensor[:, i, j] = pixel_values  # Keep as is or assign a small value if needed
                else:
                    min_val = torch.min(pixel_values)  # Min of pixel values over time
                    max_val = torch.max(pixel_values)  # Max of pixel values over time
                    
                    # Normalize the time series for the current pixel
                    normalized_tensor[:, i, j] = (pixel_values - min_val) / (max_val - min_val + small_value)  # Normalize across time
        
        return normalized_tensor
 
    def _normalize_temp_frames(self, tensor):
        normalized_tensor = torch.empty_like(tensor)
        for i in range(tensor.shape[0]):  # Loop over each frame
            # Create mask for non-zero values
            mask = tensor[i] != 0
            if mask.sum() == 0:
                normalized_tensor[i] = tensor[i]  # If all values are zero, keep as is
            else:
                min_val = torch.min(tensor[i][mask])  # Min of non-zero values
                max_val = torch.max(tensor[i][mask])  # Max of non-zero values
                # Normalize non-zero values, keep zeros unchanged
                normalized_tensor[i] = tensor[i]
                normalized_tensor[i][mask] = (tensor[i][mask] - min_val) / (max_val - min_val)
        return normalized_tensor
    
    def _normalize_labels(self, tensor):
        normalized_tensor = torch.empty_like(tensor)
        for i in range(tensor.shape[1]):  # Loop over each label (column)
            min_val = torch.min(tensor[:, i])
            max_val = torch.max(tensor[:, i])
            normalized_tensor[:, i] = (tensor[:, i] - min_val) / (max_val - min_val)
        return normalized_tensor
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        ppt_sample = self.ppt[idx]
        tmin_sample = self.tmin[idx]
        tmax_sample = self.tmax[idx]
        label = self.labels[idx]
        return {'ppt': ppt_sample, 'tmin': tmin_sample, 'tmax': tmax_sample, 'label': label}


    
    def plot_label_distribution(self, save_dir):
        """
        This function plots the distribution of each label separately and saves the plots as .png files.
        
        Parameters:
        - save_dir: Directory where the plots will be saved.
        """
        labels_np = self.labels.numpy()

        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        for i in range(labels_np.shape[1]):  # Loop through each label (column)
            plt.figure(figsize=(10, 6))
            sns.histplot(labels_np[:, i], bins=50, kde=True, color='blue')
            plt.title(f'Distribution of Label {i + 1}')
            plt.xlabel('Label Value')
            plt.ylabel('Frequency')
            plt.grid(True)

            # Save each plot as a separate .png file
            filename = f'label_{i + 1}_distribution_normalization.png'
            save_path = os.path.join(save_dir, filename)
            plt.savefig(save_path)
            plt.close()

            print(f'Label {i + 1} distribution plot saved to {save_path}')

    
    
    def plot_label_continuous(self, save_dir='/home/talhamuh/water-research/CNN-LSMT/src/cnn_lstm_project/data_plots/original_continuous'):
        """
        This function plots a continuous line plot for each label separately and saves the plots as .png files.
        
        Parameters:
        - save_dir: Directory where the plots will be saved.
        """
        labels_np = self.labels.numpy()

        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        for i in range(labels_np.shape[1]):  # Loop through each label (column)
            plt.figure(figsize=(10, 6))
            plt.plot(range(len(labels_np)), labels_np[:, i], color='blue', alpha=0.7)
            plt.title(f'Continuous Plot of Label {i + 1}')
            plt.xlabel('Index')
            plt.ylabel(f'Label {i + 1} Value')
            plt.grid(True)

            # Save each plot as a separate .png file
            filename = f'label_{i + 1}_continuous.png'
            save_path = os.path.join(save_dir, filename)
            plt.savefig(save_path)
            plt.close()

            print(f'Label {i + 1} continuous plot saved to {save_path}')


    def plot_scatter_flattened_2d(self, data, data_name, save_dir='/home/talhamuh/water-research/CNN-LSMT/src/cnn_lstm_project/data_plots/original_continuous'):
        """
        This function plots a scatter plot with flattened 2D data for all days.
        
        Parameters:
        - data: 3D tensor of shape [days, height, width] (e.g., ppt, tmin, tmax)
        - data_name: Name of the data being plotted (ppt, tmin, tmax) to use in titles and filenames.
        - save_dir: Directory where the plots will be saved.
        """
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        days = data.shape[0]
        
        # Initialize lists to hold all the x and y values
        x_values = []
        y_values = []
        
        # Collect flattened 2D data and corresponding days
        for day in range(10):
            # Flatten the 2D array for the current day
            flattened_values = data[day].flatten()
            # Append the current day index for each flattened value
            x_values.extend([day] * len(flattened_values))
            # Append the flattened values
            y_values.extend(flattened_values)

        # Create a single scatter plot for all days
        plt.figure(figsize=(12, 8))
        plt.scatter(x_values, y_values, color='blue', alpha=0.5, s=1)  # s=1 makes the points smaller
        plt.title(f'Scatter Plot of Flattened {data_name} for All Days')
        plt.xlabel('Day')
        plt.ylabel(f'{data_name} Value')
        plt.grid(True)

        # Save the plot as a .png file
        filename = f'{data_name}_all_days_scatter.png'
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path)
        plt.close()

        print(f'{data_name} scatter plot for all days saved to {save_path}')
