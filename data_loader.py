import h5py
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class HDF5Dataset(Dataset):
    def __init__(self, file_path, variables, labels_path, start_year, end_year):
        self.file_path = file_path
        self.variables = variables
        self.start_year = start_year
        self.end_year = end_year
        self.years = list(range(start_year, end_year + 1))
        
        # Load labels and convert to float, handling non-numeric values
        labels_df = pd.read_csv(labels_path)
        # labels_df = labels_df.iloc[:, 1:55]
        labels_df = labels_df.apply(pd.to_numeric, errors='coerce')  # Convert non-numeric to NaN
        labels_df.fillna(0, inplace=True)  # Replace NaNs with 0
        # Split labels into training and testing
        labels_df = labels_df.iloc[:, 1:]
        self.labels = self.normalize_labels(labels_df)
        
        
        # Determine the total number of days available
        self.total_days = 0
        with h5py.File(file_path, 'r') as file:
            for year in self.years:
                self.total_days += file[variables[0]][str(year)].shape[0]
        
        # Variables to hold global min and max values for normalization
        self.global_min = {var: float('inf') for var in variables}
        self.global_max = {var: float('-inf') for var in variables}
        self.calculate_global_min_max()
        
    def calculate_global_min_max(self):
        with h5py.File(self.file_path, 'r') as file:
            for year in self.years:
                for var in self.variables:
                    data = file[var][str(year)][:]
                    valid_data = data[np.isfinite(data)]  # Consider only finite values
                    if valid_data.size > 0:
                        self.global_min[var] = min(self.global_min[var], np.nanmin(valid_data))
                        self.global_max[var] = max(self.global_max[var], np.nanmax(valid_data))
                        
    
    def normalize_labels(self, labels_df):
        # Ensure all values are positive by shifting the dataset
        labels_df += 1 - labels_df.min().clip(upper=0)

        # Apply log transformation
        transformed_labels = np.log1p(labels_df)

        # Calculate min/max for each column of the transformed data
        label_min = transformed_labels.min()
        label_max = transformed_labels.max()
        
        # Avoid division by zero in normalization
        label_range = np.where((label_max - label_min) == 0, 1, (label_max - label_min))
        
        # Normalize and convert to numpy array for faster access
        return ((transformed_labels - label_min) / label_range).values.astype(np.float32)

    
    def _normalize_globally(self, tensor, var, idx):
        # Retrieve global min and max for the variable
        overall_min = self.global_min[var]
        overall_max = self.global_max[var]
        
        normalized_tensor = torch.empty_like(tensor)
        
        small_value = 1e-6
        
        # Normalize using the overall min/max
        if torch.all(tensor == 0):
            normalized_tensor.fill_(small_value)
        else:
            mask = tensor != 0
            normalized_tensor[mask] = (tensor[mask] - overall_min) / (overall_max - overall_min)
            normalized_tensor[~mask] = small_value  # Assign small_value to zero elements if any
       
        return normalized_tensor

    def __len__(self):
        return self.total_days

    def __getitem__(self, idx):
        data = {}
        with h5py.File(self.file_path, 'r') as file:
            cumulative_days = 0
            target_year = None
            day_index_in_year = None

            for year in self.years:
                num_days_this_year = file[self.variables[0]][str(year)].shape[0]
                if idx < cumulative_days + num_days_this_year:
                    target_year = year
                    day_index_in_year = idx - cumulative_days
                    break
                cumulative_days += num_days_this_year

            if target_year is not None:
                for var in self.variables:
                    daily_data = file[var][str(target_year)][day_index_in_year]
                    # Convert NaNs to zero
                    daily_data = np.nan_to_num(daily_data, nan=0)
                    tensor_data = torch.tensor(daily_data, dtype=torch.float32)
                    data[var] = self._normalize_globally(tensor_data, var, idx)

        # Ensure the label index is valid
        if idx < len(self.labels):
            data['label'] = torch.tensor(self.labels[idx], dtype=torch.float32)
        else:
            data['label'] = torch.tensor(0, dtype=torch.float32)  # Default or error handling

        return data
