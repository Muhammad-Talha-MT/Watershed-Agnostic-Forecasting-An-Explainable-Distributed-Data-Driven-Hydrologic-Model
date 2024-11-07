import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from data_loader import read_hdf5_data_parallel
from dataset import ClimateDataset
from model import CNN_LSTM, ResNet_LSTM
import pandas as pd
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from visualize import visualize_top_examples, visualize_box_plot
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import argparse

def calculate_nse(y_true, y_pred):
    mean_observed = torch.mean(y_true, dim=0)
    numerator = torch.sum((y_true - y_pred) ** 2, dim=0)
    denominator = torch.sum((y_true - mean_observed) ** 2, dim=0)
    nse = 1 - (numerator / denominator)
    return nse


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    start_epoch = checkpoint['epoch']
    
    if optimizer and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    if scheduler and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])
    
    print(f"Checkpoint loaded: {checkpoint_path}")
    print(f"Starting from epoch: {start_epoch + 1}")
    
    first_param_name, first_param_tensor = next(iter(model.named_parameters()))
    print(f"First parameter after loading checkpoint: {first_param_name}, {first_param_tensor[0,0]}")

    if optimizer:
        print(f"Learning rate after loading checkpoint: {optimizer.param_groups[0]['lr']}")
    
    return model, optimizer, scheduler, start_epoch

def plot_predictions(model, dataloader, device, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    model.eval()
    all_outputs = []
    all_labels = []

    with torch.no_grad():
        for data in dataloader:
            ppt = data['ppt'].to(device)
            tmin = data['tmin'].to(device)
            tmax = data['tmax'].to(device)
            labels = data['label'].to(device)
            
            outputs = model(ppt, tmin, tmax)
            all_outputs.append(outputs)
            all_labels.append(labels)
        
    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    outputs = all_outputs.cpu().numpy()
    labels = all_labels.cpu().numpy()
    
    mse_values = []
    nse_values = []

    for i in range(outputs.shape[1]):
        output = outputs[:, i]
        label = labels[:, i]
        mse = mean_squared_error(label, output)
        nse = calculate_nse(torch.tensor(label), torch.tensor(output)).item()

        mse_values.append(mse)
        nse_values.append(nse)

        r2 = r2_score(label, output)
    
        print(f'Training Data - R²: {r2:.2f}, NSE: {nse:.2f}, MSE: {mse:.2f}')
        
        plt.figure(figsize=(10, 6))
        plt.scatter(label, output, alpha=0.6, color='blue', label='Training Data')
        plt.xlabel('Actual Streamflow')
        plt.ylabel('Predicted Streamflow')
        plt.title('Comparison of Actual and Predicted Streamflow')
        plt.plot([label.min(), label.max()], [label.min(), label.max()], 'k--')
        plt.grid(True)
        plt.legend()
        
        plt.text(0.05, 0.95, f'Training Data - R²: {r2:.2f}', transform=plt.gca().transAxes)
        plt.text(0.05, 0.90, f'Training Data - NSE: {nse:.2f}', transform=plt.gca().transAxes)
        plt.text(0.05, 0.85, f'Training Data - RMSE: {mse:.2f}', transform=plt.gca().transAxes)
        plt.savefig(os.path.join(save_dir, f'output_{i+1}_700.png'))

def main(args):
    h5_file = args.h5_file
    checkpoint_path = args.checkpoint_path
    labels_path = args.labels_path
    save_dir = args.save_dir

    ppt = read_hdf5_data_parallel(h5_file, 'ppt', 2000, 2004)
    tmin = read_hdf5_data_parallel(h5_file, 'tmin', 2000, 2004)
    tmax = read_hdf5_data_parallel(h5_file, 'tmax', 2000, 2004)

    labels = pd.read_csv(labels_path)
    labels = labels[:1825]
    logging.info('Dataset loaded')
    
    min_length = min(len(ppt), len(tmin), len(tmax), len(labels))
    min_length = 1825
    ppt = ppt[:min_length]
    tmin = tmin[:min_length]
    tmax = tmax[:min_length]
    labels = labels[:min_length]
    
    dataset = ClimateDataset(ppt, tmin, tmax, labels)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    model = ResNet_LSTM()  # or 'resnet50' for deeper ResNet
    model, _, _, _ = load_checkpoint(checkpoint_path, model)
    
    if torch.cuda.is_available():
        available_gpus = torch.cuda.device_count()
        logging.info(f"Available GPUs: {available_gpus}")
        device_ids = list(range(available_gpus))
        logging.info(f"Using device IDs: {device_ids}")
        model.to(device_ids[0])
        device = torch.device(f"cuda:{device_ids[0]}")
        logging.info(f"Using device: {device}")
    else:
        device = torch.device("cpu")
        logging.info(f"Using device: {device}")

    plot_predictions(model, dataloader, device, save_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Climate Data Analysis with CNN-LSTM and ResNet-LSTM')
    parser.add_argument('--h5_file', type=str, required=True, help='Path to the HDF5 file')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--labels_path', type=str, required=True, help='Path to the CSV file with labels')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save plots')

    args = parser.parse_args()
    main(args)
