# data_loader.py

import h5py
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def read_year_data(f, variable, year):
    return f[variable][str(year)][:]

def read_hdf5_data_parallel(file_path, variable, start_year, end_year):
    years = range(start_year, end_year + 1)
    data_list = []

    with h5py.File(file_path, 'r') as f:
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(read_year_data, f, variable, year) for year in years]
            for future in futures:
                data_list.append(future.result())

    return np.concatenate(data_list, axis=0)

