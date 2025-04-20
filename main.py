import os
import yaml
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'
import torch
from torchsummary import summary
from torch.utils.data import DataLoader, random_split, Dataset, Subset
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from torch.utils.tensorboard import SummaryWriter
from data_loader import HDF5Dataset
from dataset import ClimateDataset
from model import ResNet_LSTM, CNN_LSTM
import pandas as pd
from visualize import visualize_all_examples, visualize_label_distributions, visualize_all_examples_seq
from sklearn.metrics import r2_score
import logging
from logging.config import dictConfig  # Ensure this import is present
# Set print options to display all data
# torch.set_printoptions(profile="full")
# Load logging configuration from a YAML file
def setup_logging(path='config/logging_config.yaml'):
    with open(path, 'r') as file:
        config = yaml.safe_load(file.read())
        logging.config.dictConfig(config)


def save_checkpoint(state, filename):
    torch.save(state, filename)


def load_checkpoint(checkpoint_path, model, optimizer, device):
    # checkpoint = torch.load(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    # Retrieve the state_dict from the checkpoint
    state_dict = checkpoint['state_dict']
    
    # Check if the model is in DataParallel mode (i.e., it has 'module.' prefixes)
    if isinstance(model, nn.DataParallel):
        # If loading a single-GPU model checkpoint, add 'module.' prefix
        state_dict = {("module." + k if not k.startswith("module.") else k): v for k, v in state_dict.items()}
    else:
        # If model is not in DataParallel mode, ensure there are no 'module.' prefixes
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    
    model.to(device)
    optimizer.load_state_dict(checkpoint['optimizer'])
     
    start_epoch = checkpoint['epoch'] + 1
    
    return model, optimizer, None, start_epoch  # Return None for scheduler if not in checkpoint



def calculate_nse(y_true, y_pred):
    """Calculate the Nash-Sutcliffe Efficiency."""
    mean_observed = torch.mean(y_true, dim=0)
    
    numerator = torch.sum((y_true - y_pred) ** 2, dim=0)
    denominator = torch.sum((y_true - mean_observed) ** 2, dim=0)
    # Handle division by zero by using a small epsilon
    
    nse = 1 - (numerator / denominator)
    # if batch_idx >= 52:
    #     logger.warning('y_true: {}, mean_observed: {}'.format(y_true, mean_observed))
    #     logger.warning('batch_id: {}, Num: {}, Denm: {}, nse {}'.format(batch_idx, numerator, denominator, nse))
    return nse.mean()

# Train the model
def train_model(model, train_loader, optimizer, criterion, device, writer, epoch, logger):
    model.train()
    total_loss = 0.0
    total_nse = 0.0
    total_mse = 0.0
    total_samples = 0  # Keep track of total number of samples processed

    for batch_idx, data in enumerate(train_loader):
        ppt = data['ppt'].to(device)
        tmin = data['tmin'].to(device)
        tmax = data['tmax'].to(device)
        labels = data['label'].to(device)
        
        # Getting the actual batch size (useful for the last batch)
        current_batch_size = labels.size(0)
        
        optimizer.zero_grad()
        outputs = model(ppt, tmin, tmax)
        # outputs = outputs[:, :54]  # Assuming you want the first 54 labels for training
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Accumulate the loss and metrics, weighted by batch size
        total_loss += loss.item() * current_batch_size
        nse = calculate_nse(labels, outputs)
        
        total_nse += nse.item() * current_batch_size
        logger.debug('batch_id: {}, nse: {}, loss: {}'.format(batch_idx, nse.item(), loss))
        mse = torch.mean((outputs - labels) ** 2).item()
        total_mse += mse * current_batch_size
        
        # Update total samples
        total_samples += current_batch_size

    # Calculate the averages using the total samples
    avg_loss = total_loss / total_samples
    avg_nse = total_nse / total_samples
    avg_mse = total_mse / total_samples  # Average MSE
    return avg_loss, avg_nse, avg_mse


def val_model(model, val_loader, criterion, device, writer, epoch):
    model.eval()
    total_loss = 0.0
    total_nse = 0.0
    total_mse = 0.0
    total_samples = 0  # Keep track of the total number of samples processed

    with torch.no_grad():
        for batch_idx, data in enumerate(val_loader):
            ppt = data['ppt'].to(device)
            tmin = data['tmin'].to(device)
            tmax = data['tmax'].to(device)
            labels = data['label'].to(device)
            
            outputs = model(ppt, tmin, tmax)
            # outputs = outputs[:, 54:64]  # Assuming these are the correct indices for validation

            # Calculate loss for the current batch
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)  # Weight by the batch size

            # Calculate NSE and MSE for the current batch, weighted
            nse = calculate_nse(labels, outputs)
            total_nse += nse.item() * labels.size(0)
            mse = torch.mean((outputs - labels) ** 2).item()
            total_mse += mse * labels.size(0)

            # Update total samples count
            total_samples += labels.size(0)

    # Calculate average loss, NSE, and MSE over all batches
    avg_loss = total_loss / total_samples
    avg_nse = total_nse / total_samples
    avg_mse = total_mse / total_samples  # Average MSE

    return avg_loss, avg_nse, avg_mse


def inference(model, dataloader, device, save_dir, logger):
    model.eval()
    all_outputs = []
    all_labels = []
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            ppt = data['ppt'].to(device)
            tmin = data['tmin'].to(device)
            tmax = data['tmax'].to(device)
            labels = data['label'].to(device)
            outputs = model(ppt, tmin, tmax)
            # outputs = outputs[:, 54:64]  # Adjust index if needed
            # outputs = outputs[:, :54]

            all_outputs.append(outputs)
            all_labels.append(labels)

    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    mse_values = []
    nse_values = []
    r2_values = []
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for i in range(all_outputs.shape[1]):
        output = all_outputs[:, i]
        label = all_labels[:, i]

        # Convert tensors to CPU for matplotlib compatibility
        output_cpu = output.cpu().numpy()
        label_cpu = label.cpu().numpy()
        time_index = range(len(output_cpu))  # Time series index
        
        mse = torch.mean((output - label) ** 2).item()
        nse = calculate_nse(label, output).item()
        r2 = r2_score(label_cpu, output_cpu)

        mse_values.append(mse)
        nse_values.append(nse)
        r2_values.append(r2)

        # logger.info(f'Label {i+1} - R²: {r2:.2f}, NSE: {nse:.2f}, MSE: {mse:.2f}')
        
        # plt.figure(figsize=(10, 6))
        # plt.scatter(label_cpu, output_cpu, alpha=0.6, color='blue', label='Predictions')
        # plt.plot([label_cpu.min(), label_cpu.max()], [label_cpu.min(), label_cpu.max()], 'k--', lw=2, label='Ideal Fit')
        # plt.xlabel('Actual Values')
        # plt.ylabel('Predicted Values')
        # plt.title(f'Actual vs. Predicted - Label {i+1}')
        # plt.grid(True)
        # plt.legend()  # This will now correctly display the legend
        # plt.text(0.05, 0.95, f'R²: {r2:.2f}', transform=plt.gca().transAxes)
        # plt.text(0.05, 0.90, f'NSE: {nse:.2f}', transform=plt.gca().transAxes)
        # plt.text(0.05, 0.85, f'MSE: {mse:.2f}', transform=plt.gca().transAxes)
        # plt.savefig(os.path.join(save_dir, f'output_{i}.png'))
        # plt.close()
        logger.info(f'Label {i+1} - R²: {r2:.2f}, NSE: {nse:.2f}, MSE: {mse:.2f}')
        
        # Plot Time Series
        plt.figure(figsize=(12, 6))
        plt.plot(time_index, label_cpu, label='Actual', color='blue', linewidth=2)
        plt.plot(time_index, output_cpu, label='Predicted', color='red', linestyle='dashed', linewidth=2)
        plt.xlabel('Time')
        plt.ylabel('Streamflow')
        plt.title(f'Time Series Plot - Label {i+1}')
        plt.legend()
        plt.grid(True)

        # Annotate with performance metrics
        plt.text(0.05, 0.90, f'R²: {r2:.2f}', transform=plt.gca().transAxes)
        plt.text(0.05, 0.85, f'NSE: {nse:.2f}', transform=plt.gca().transAxes)
        plt.text(0.05, 0.80, f'MSE: {mse:.2f}', transform=plt.gca().transAxes)

        plt.savefig(os.path.join(save_dir, f'timeseries_output_{i}.png'))
        plt.close()
        
        
        # === Percent Error Histogram Plot ===
        percent_error = 100 * (output_cpu - label_cpu) / (label_cpu + 1e-8)
        hist_colors = ['red' if e > 0 else 'blue' for e in percent_error]

        plt.figure(figsize=(12, 6))
        plt.bar(time_index, percent_error, color=hist_colors, alpha=0.7)
        plt.axhline(0, color='black', linewidth=1)
        plt.title(f'Percent Error Histogram - Label {i+1}')
        plt.xlabel('Days')
        plt.ylabel('Percent Error (%)')
        plt.grid(True)

        hist_dir = os.path.join(save_dir, 'error_histograms')
        os.makedirs(hist_dir, exist_ok=True)
        plt.savefig(os.path.join(hist_dir, f'percent_error_histogram_label_{i+1}.png'))
        plt.close()



def load_yaml_config(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


def main():
    # Load config from YAML file
    config = load_yaml_config('config/config.yaml')
        # Set up logging
    setup_logging()
    
    # Get the logger specified in the YAML file
    logger = logging.getLogger('my_application')
    # Set primary device for DataParallel
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

    # Set up TensorBoard writer
    writer = SummaryWriter(log_dir=config['tensorboard_logdir'])

    # Load data
    variables_to_load = ['ppt', 'tmin', 'tmax']
    dataset = HDF5Dataset(config['h5_file'], variables_to_load, config['labels_path'], 2000, 2009)
    dataset_size = len(dataset) 
    # dataset = HDF5Dataset(config['h5_file'], variables_to_load, config['labels_path'], 2010, 2019)
    # loader = DataLoader(dataset, batch_size=config['batch_size'], num_workers=32, shuffle=False)
    # print("yes")
    # visualize_label_distributions(loader, 61, '/home/talhamuh/water-research/CNN-LSMT/src/cnn_lstm_project/data_plots/min-max-O-4y')
    # exit()
    # for id, data in enumerate(loader):
    #     print(data['ppt'].shape)
    #     print(data['label'].shape)
    #     exit()
    # visualize_all_examples(loader, 5, "/home/talhamuh/water-research/CNN-LSMT/src/cnn_lstm_project/data_plots/first_100_global_optimized_dataloader")
    # visualize_all_examples_seq(loader, batch_index=5, save_dir='/home/talhamuh/water-research/CNN-LSMT/src/cnn_lstm_project/data_plots/first_batch_seq')
    # exit()
    # val_dataset = HDF5Dataset(config['h5_file'], variables_to_load, config['labels_path'], 2007, 2009)
    # test_dataset = HDF5Dataset(config['h5_file'], variables_to_load, config['labels_path'], 2010, 2010)
    # train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], num_workers=32, shuffle=False)
    # val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], num_workers=32, shuffle=False)
    # test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], num_workers=32, shuffle=False)
    # Define the split sizes
    num_train = int(0.8 * len(dataset))
    num_val = int(0.1 * len(dataset))
    num_test = len(dataset) - num_train - num_val
    # Randomly split the dataset
    # train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [num_train, num_val, num_test])
    
    
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
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=32)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=32)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=32)
    
    
    # test_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False, num_workers=32)
    # visualize_all_examples(train_loader, 16, "/home/talhamuh/water-research/CNN-LSMT/src/cnn_lstm_project/data_plots/first_100_global_optimized_dataloader")
    # Initialize model, optimizer, and loss function
    model = CNN_LSTM().to(device)
    start_epoch = 0
    model = nn.DataParallel(model, device_ids=[0, 1, 2])  # Multi-GPU support with DataParallel
    # Freeze the CNN and LSTM layers
    
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    criterion = nn.MSELoss()
    if config['mode'] == 'infer':
        model, optimizer, scheduler, start_epoch = load_checkpoint(config['checkpoint_path'], model, optimizer, device)
        # Define batch size and input dimensions
        # batch_size = 1  # Adjust as needed
        # height, width = 1849, 1458  # Replace with your actual spatial dimensions

        # Create dummy inputs for ppt, tmin, tmax
        # ppt_dummy = torch.randn(batch_size, height, width)
        # tmin_dummy = torch.randn(batch_size, height, width)
        # tmax_dummy = torch.randn(batch_size, height, width)
        # writer.add_graph(model, (ppt_dummy, tmin_dummy, tmax_dummy))
        # inference_loader = DataLoader(val_loader, batch_size=config['batch_size'])
        inference(model, test_loader, device, 'results/03212025/test', logger)
    if config['mode'] == 'train' :
        if config['resume']:
            model, optimizer, scheduler, start_epoch = load_checkpoint(config['checkpoint_path'], model, optimizer, device)
        # if config['fine_tune']:
        #     model.module.freeze_backbone() if isinstance(model, nn.DataParallel) else model.freeze_backbone()
        #     optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
            
        for epoch in range(start_epoch, config['epochs']):
            train_loss, train_nse, train_mse = train_model(model, train_loader, optimizer, criterion, device, writer, epoch, logger)
            writer.add_scalar('Loss/Train', train_loss, epoch)
            writer.add_scalar('NSE/Train', train_nse, epoch)
            writer.add_scalar('MSE/Train', train_mse, epoch)
            print(f"Epoch [{epoch+1}/{config['epochs']}], Train Loss: {train_loss:.4f}, Train NSE: {train_nse:.4f}, Train MSE: {train_mse:.4f}")

            # Unfreeze LSTM after 10 epochs
            # if epoch == start_epoch + 10 and config['fine_tune']:
            #     for name, param in model.named_parameters():
            #         if "lstm" in name:
            #             param.requires_grad = True
            #     optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5)

            # # Unfreeze CNN after 20 epochs
            # if epoch == start_epoch + 20 and config['fine_tune']:
            #     for name, param in model.named_parameters():
            #         param.requires_grad = True
            #     optimizer = optim.Adam(model.parameters(), lr=1e-5)     
                
                       
            if (epoch + 1) % 10 == 0:
                val_loss, val_nse, val_mse = val_model(model, val_loader, criterion, device, writer, epoch)
                writer.add_scalar('Loss/val', val_loss, epoch)
                writer.add_scalar('NSE/val', val_nse, epoch)
                writer.add_scalar('MSE/val', val_mse, epoch)
                print(f"Epoch [{epoch+1}/{config['epochs']}], val Loss: {val_loss:.4f}, val NSE: {val_nse:.4f}, Train MSE: {val_mse:.4f}")
            if (epoch + 1) % 50 == 0:
                save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, filename=config['save_checkpoint_path'])
    writer.close()

if __name__ == "__main__":
    main()
