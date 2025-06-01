import h5py
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class HDF5Dataset(Dataset):
    def __init__(self, file_path, variables, labels_path, start_year, end_year, sequence_length=5):
        self.file_path = file_path
        self.variables = variables
        self.sequence_length = sequence_length
        self.start_year = start_year
        self.end_year = end_year
        self.years = list(range(start_year, end_year + 1))

        # # Load and preprocess labels
        # labels_df = pd.read_csv(labels_path)
        # labels_df = labels_df.apply(pd.to_numeric, errors='coerce')
        # labels_df.fillna(0, inplace=True)
        # self.labels = labels_df.iloc[:, 1:].values.astype(np.float32)  # Adjust as needed

        labels_df = pd.read_csv(labels_path)
        # Convert the first column to datetime (assuming it is named 'date')
        labels_df['date'] = pd.to_datetime(labels_df['date'], errors='coerce')
        # Filter rows based on the year
        labels_df = labels_df[(labels_df['date'].dt.year >= start_year) & (labels_df['date'].dt.year <= end_year)]
        # Drop the date column and convert remaining values to float32
        labels_df = labels_df.drop(columns=['date'])
        labels_df = labels_df.apply(pd.to_numeric, errors='coerce')
        labels_df.fillna(0, inplace=True)
        self.labels = labels_df.values.astype(np.float32)
        # self.labels = self.normalize_labels(self.labels)

        # Determine the total number of days
        self.total_days = 0
        with h5py.File(file_path, 'r') as file:
            for year in self.years:
                self.total_days += file[variables[0]][str(year)].shape[0]
        self.total_days -= (self.sequence_length - 1)  # Adjust for sequences

        # Min/max for normalization
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
        # labels_df += 1 - labels_df.min().clip(upper=0)
        # labels_df += 1 - np.clip(labels_df.min(), None, 0)
        # Apply log transformation
        # transformed_labels = np.log1p(labels_df)

        # Calculate min/max for each column of the transformed data
        label_min = labels_df.min()
        label_max = labels_df.max()
        
        # Avoid division by zero in normalization
        label_range = np.where((label_max - label_min) == 0, 1, (label_max - label_min))
        
        # Normalize and convert to numpy array for faster access
        return ((labels_df - label_min) / label_range).astype(np.float32)

    
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

    def _normalize_labels(self, tensor):
        """
        Normalize each column (label) of the input tensor to the [0, 1] range.

        Parameters
        ----------
        tensor : torch.Tensor
            A 2D tensor of shape (N, num_labels) representing raw labels.

        Returns
        -------
        torch.Tensor
            A tensor of the same shape with values normalized per column.
        """
        normalized_tensor = torch.empty_like(tensor)
        for i in range(tensor.shape[1]):  # Loop over each label (column)
            min_val = torch.min(tensor[:, i])
            max_val = torch.max(tensor[:, i])
            # Avoid division by zero if the label is constant
            if max_val - min_val == 0:
                normalized_tensor[:, i] = 0.0
            else:
                normalized_tensor[:, i] = (tensor[:, i] - min_val) / (max_val - min_val)
        return normalized_tensor

    def __len__(self):
        return self.total_days

    def __getitem__(self, idx):
        data_sequences = {var: [] for var in self.variables}
        label = None  # Single label for the 5th day

        with h5py.File(self.file_path, 'r') as file:
            start_idx = idx
            end_idx = idx + self.sequence_length

            # Adjust start and end indices if the end index goes beyond the available data
            if end_idx > self.total_days:
                end_idx = self.total_days
                start_idx = end_idx - self.sequence_length  # Shift the start index back to maintain sequence length

            for day_offset in range(self.sequence_length):
                day_idx = start_idx + day_offset
                year, day_in_year = self.find_year_and_day(day_idx, file)

                for var in self.variables:
                    daily_data = file[var][str(year)][day_in_year]
                    daily_data = np.nan_to_num(daily_data, nan=0)  # Replace NaN with 0
                    tensor_data = torch.tensor(daily_data, dtype=torch.float32)
                    normalized_data = self._normalize_globally(tensor_data, var, idx)
                    data_sequences[var].append(normalized_data)

                # Fetch the label only for the 5th day (last day in the sequence)
                if day_offset == self.sequence_length - 1:
                    label = torch.tensor(self.labels[day_idx], dtype=torch.float32)

        # Stack sequences along a new dimension (time dimension)
        for var in data_sequences:
            data_sequences[var] = torch.stack(data_sequences[var], dim=0)

        # Return the sequences and the label for the 5th day
        return {**data_sequences, 'label': label}


    def find_year_and_day(self, idx, file):
        cumulative_days = 0
        for year in self.years:
            num_days_this_year = file[self.variables[0]][str(year)].shape[0]
            if idx < cumulative_days + num_days_this_year:
                return year, idx - cumulative_days
            cumulative_days += num_days_this_year
        raise IndexError("Day index out of dataset range")

# import h5py
# import torch
# import numpy as np
# import pandas as pd
# from torch.utils.data import Dataset, DataLoader

# class HDF5Dataset(Dataset):
#     def __init__(self, file_path, variables, labels_path, start_year, end_year):
#         self.file_path = file_path
#         self.variables = variables
#         self.start_year = start_year
#         self.end_year = end_year
#         self.years = list(range(start_year, end_year + 1))
        
#         # Load labels and convert to float, handling non-numeric values
#         labels_df = pd.read_csv(labels_path)
#         # labels_df = labels_df.iloc[:, 1:55]
#         labels_df = labels_df.apply(pd.to_numeric, errors='coerce')  # Convert non-numeric to NaN
#         labels_df.fillna(0, inplace=True)  # Replace NaNs with 0
#         # Split labels into training and testing
#         labels_df = labels_df.iloc[:, 1:]
#         self.labels = self.normalize_labels(labels_df)
        
        
#         # Determine the total number of days available
#         self.total_days = 0
#         with h5py.File(file_path, 'r') as file:
#             for year in self.years:
#                 self.total_days += file[variables[0]][str(year)].shape[0]
        
#         # Variables to hold global min and max values for normalization
#         self.global_min = {var: float('inf') for var in variables}
#         self.global_max = {var: float('-inf') for var in variables}
#         self.calculate_global_min_max()
        
#     def calculate_global_min_max(self):
#         with h5py.File(self.file_path, 'r') as file:
#             for year in self.years:
#                 for var in self.variables:
#                     data = file[var][str(year)][:]
#                     valid_data = data[np.isfinite(data)]  # Consider only finite values
#                     if valid_data.size > 0:
#                         self.global_min[var] = min(self.global_min[var], np.nanmin(valid_data))
#                         self.global_max[var] = max(self.global_max[var], np.nanmax(valid_data))
                        
    
#     def normalize_labels(self, labels_df):
#         # Ensure all values are positive by shifting the dataset
#         labels_df += 1 - labels_df.min().clip(upper=0)

#         # Apply log transformation
#         transformed_labels = np.log1p(labels_df)

#         # Calculate min/max for each column of the transformed data
#         label_min = transformed_labels.min()
#         label_max = transformed_labels.max()
        
#         # Avoid division by zero in normalization
#         label_range = np.where((label_max - label_min) == 0, 1, (label_max - label_min))
        
#         # Normalize and convert to numpy array for faster access
#         return ((transformed_labels - label_min) / label_range).values.astype(np.float32)

    
#     def _normalize_globally(self, tensor, var, idx):
#         # Retrieve global min and max for the variable
#         overall_min = self.global_min[var]
#         overall_max = self.global_max[var]
        
#         normalized_tensor = torch.empty_like(tensor)
        
#         small_value = 1e-6
        
#         # Normalize using the overall min/max
#         if torch.all(tensor == 0):
#             normalized_tensor.fill_(small_value)
#         else:
#             mask = tensor != 0
#             normalized_tensor[mask] = (tensor[mask] - overall_min) / (overall_max - overall_min)
#             normalized_tensor[~mask] = small_value  # Assign small_value to zero elements if any
       
#         return normalized_tensor

#     def __len__(self):
#         return self.total_days

#     def __getitem__(self, idx):
#         data = {}
#         with h5py.File(self.file_path, 'r') as file:
#             cumulative_days = 0
#             target_year = None
#             day_index_in_year = None

#             for year in self.years:
#                 num_days_this_year = file[self.variables[0]][str(year)].shape[0]
#                 if idx < cumulative_days + num_days_this_year:
#                     target_year = year
#                     day_index_in_year = idx - cumulative_days
#                     break
#                 cumulative_days += num_days_this_year

#             if target_year is not None:
#                 for var in self.variables:
#                     daily_data = file[var][str(target_year)][day_index_in_year]
#                     # Convert NaNs to zero
#                     daily_data = np.nan_to_num(daily_data, nan=0)
#                     tensor_data = torch.tensor(daily_data, dtype=torch.float32)
#                     data[var] = self._normalize_globally(tensor_data, var, idx)

#         # Ensure the label index is valid
#         if idx < len(self.labels):
#             data['label'] = torch.tensor(self.labels[idx], dtype=torch.float32)
#         else:
#             data['label'] = torch.tensor(0, dtype=torch.float32)  # Default or error handling

#         return data
