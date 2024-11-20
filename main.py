import os
import yaml
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
import torch
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
import argparse
from visualize import visualize_all_examples, visualize_label_distributions

def save_checkpoint(state, filename):
    torch.save(state, filename)

# def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
#     checkpoint = torch.load(checkpoint_path)
#     model.load_state_dict(checkpoint['state_dict'])

#     # # Filter out the weights for the fully connected layer
#     # state_dict = checkpoint['state_dict']
#     # # Remove the last layer weights (fc3) to avoid shape mismatch
#     # state_dict = {k: v for k, v in state_dict.items() if 'fc3' not in k}
    
#     # # Load the remaining state_dict into the model
#     # model.load_state_dict(state_dict, strict=False)  # strict=False allows missing key
    
#     start_epoch = checkpoint['epoch']
    
#     if optimizer and scheduler:
#         optimizer.load_state_dict(checkpoint['optimizer'])
#         scheduler.load_state_dict(checkpoint['scheduler'])
    
#     return model, optimizer, scheduler, start_epoch

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
    epsilon = 1e-10
    denominator = torch.where(denominator == 0, epsilon, denominator)

    nse = 1 - (numerator / denominator)
    return nse.mean()

# Train the model
def train_model(model, train_loader, optimizer, criterion, device, writer, epoch):
    model.train()
    total_loss = 0.0
    total_nse = 0.0
    total_mse = 0.0
    for batch_idx, data in enumerate(train_loader):
        ppt = data['ppt'].to(device)
        tmin = data['tmin'].to(device)
        tmax = data['tmax'].to(device)
        labels = data['label'].to(device)
        optimizer.zero_grad()
        outputs = model(ppt, tmin, tmax)
        outputs = outputs[:, :54]
        # print(outputs[:, :44])
        # print(outputs[:, 44:54])
        # print(outputs[:, 54:64])
        # exit()
        # print("train output:", outputs.shape)
        # print("train labels:", labels.shape)
        # Write model state_dict to the file
        # with open("/home/talhamuh/water-research/CNN-LSMT/src/cnn_lstm_project/data_plots/model_state_dict.txt", "a") as f:
        #     f.write("Batch {}\n".format(batch_idx))
        #     for key, value in model.state_dict().items():
        #         f.write(f"{key}: {value}\n")
        #     f.write("\n")  # Add some spacing between batches
        
        # # print('output:', outputs, 'labels:', labels)
        loss = criterion(outputs, labels)
        # print(f"Loss at batch {batch_idx}: {loss.item()}")
        loss.backward()
        
        # with open("/home/talhamuh/water-research/CNN-LSMT/src/cnn_lstm_project/data_plots/model_gradients.txt", "a") as grad_file:
        #     # Write gradients to the file
        #     grad_file.write(f"Batch {batch_idx}\n")
        #     for name, param in model.named_parameters():
        #         if param.grad is not None:
        #             grad_file.write(f"{name} gradients:\n{param.grad}\n")
        #         else:
        #             grad_file.write(f"{name} has no gradients\n")
        #     grad_file.write("\n")  # Add some spacing between batches
        optimizer.step()

        total_loss += loss.item()
        nse = calculate_nse(labels, outputs)
        total_nse += nse.item()
        # Calculate MSE manually
        mse = torch.mean((outputs - labels) ** 2)
        total_mse += mse.item()


    
    avg_loss = total_loss / len(train_loader)
    avg_nse = total_nse / len(train_loader)
    avg_mse = total_mse / len(train_loader)  # Average MSE
    return avg_loss, avg_nse, avg_mse

# val the model
def val_model(model, val_loader, criterion, device, writer, epoch):
    model.eval()
    total_loss = 0.0
    total_nse = 0.0
    total_mse = 0.0
    with torch.no_grad():
        for batch_idx, data in enumerate(val_loader):
            ppt = data['ppt'].to(device)
            tmin = data['tmin'].to(device)
            tmax = data['tmax'].to(device)
            labels = data['label'].to(device)
            
            outputs = model(ppt, tmin, tmax)
            outputs = outputs[:, 54:64]
            # print("val output:", outputs.shape)
            # print("val labels:", labels.shape)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            nse = calculate_nse(labels, outputs)
            total_nse += nse.item()
            # Calculate MSE manually
            mse = torch.mean((outputs - labels) ** 2)
            total_mse += mse.item()


    
    avg_loss = total_loss / len(val_loader)
    avg_nse = total_nse / len(val_loader)
    avg_mse = total_mse / len(val_loader)  # Average MSE
    return avg_loss, avg_nse, avg_mse

# Inference function
def inference(model, dataloader, device, save_dir):
    model.eval()
    all_outputs = []
    all_labels = []
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            ppt = data['ppt'].to(device)
            tmin = data['tmin'].to(device)
            tmax = data['tmax'].to(device)
            labels = data['label'].to(device)
            # Generate model predictions
            outputs = model(ppt, tmin, tmax)
            outputs = outputs[:, 54:64]
            all_outputs.append(outputs)
            all_labels.append(labels)
            
    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    outputs = all_outputs.cpu().numpy()
    labels = all_labels.cpu().numpy()
    
    mse_values = []
    nse_values = []
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for i in range(outputs.shape[1]):
        output = outputs[:, i]
        label = labels[:, i]
        
        # Calculate evaluation metrics
        mse = mean_squared_error(label, output)
        nse = calculate_nse(torch.tensor(label), torch.tensor(output)).item()
        r2 = r2_score(label, output)

        mse_values.append(mse)
        nse_values.append(nse)

        print(f'Label {i+1} - R²: {r2:.2f}, NSE: {nse:.2f}, MSE: {mse:.2f}')
        
        # Plot scatter plot for predictions
        plt.figure(figsize=(10, 6))
        plt.scatter(label, output, alpha=0.6, color='blue', label=f'Label {i+1} Data')
        plt.xlabel('Actual Streamflow')
        plt.ylabel('Predicted Streamflow')
        plt.title(f'Comparison of Actual and Predicted Streamflow - Label {i+1}')
        plt.plot([label.min(), label.max()], 
                 [label.min(), label.max()], 'k--', lw=2)  # Diagonal line
        plt.grid(True)
        plt.legend()
        
        # Add R², NSE, RMSE, and MSE to the plot for the label data
        plt.text(0.05, 0.95, f'R²: {r2:.2f}', transform=plt.gca().transAxes)
        plt.text(0.05, 0.90, f'NSE: {nse:.2f}', transform=plt.gca().transAxes)
        plt.text(0.05, 0.85, f'MSE: {mse:.2f}', transform=plt.gca().transAxes)
        
        # Save the figure
        plt.savefig(os.path.join(save_dir, f'output_{i+1}.png'))
        plt.close()


def load_yaml_config(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def main():
    # Load config from YAML file
    config = load_yaml_config('config/config.yaml')
    
    # Set primary device for DataParallel
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu") 

    # Set up TensorBoard writer
    writer = SummaryWriter(log_dir=config['tensorboard_logdir'])

    # Load data
    variables_to_load = ['ppt', 'tmin', 'tmax']
    train_dataset = HDF5Dataset(config['h5_file'], variables_to_load, config['labels_path'], 2000, 2019, mode='train')
    test_dataset = HDF5Dataset(config['h5_file'], variables_to_load, config['labels_path'], 2000, 2019, mode='test')
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], num_workers=32, shuffle=False)
    val_loader = DataLoader(test_dataset, batch_size=config['batch_size'], num_workers=32, shuffle=False)
    # for idx, batch in enumerate(train_dataloader):
    #     # Each 'batch' is a dictionary with keys corresponding to your variables and 'label'
    #     ppt = batch['ppt'].shape  # Precipitation data tensor for the batch
    #     tmax = batch['tmax'].shape  # Maximum temperature data tensor for the batch
    #     tmin = batch['tmin'].shape  # Minimum temperature data tensor for the batch
    #     labels = batch['label'].shape  # Labels tensor for the batch
    #     print(idx, ppt, tmax, tmin, labels)
        
    # for idx, batch in enumerate(test_dataloader):
    #     # Each 'batch' is a dictionary with keys corresponding to your variables and 'label'
    #     ppt = batch['ppt'].shape  # Precipitation data tensor for the batch
    #     tmax = batch['tmax'].shape  # Maximum temperature data tensor for the batch
    #     tmin = batch['tmin'].shape  # Minimum temperature data tensor for the batch
    #     labels = batch['label'].shape  # Labels tensor for the batch
    #     print(idx, ppt, tmax, tmin, labels)
    # visualize_all_examples(train_dataloader, '/home/talhamuh/water-research/CNN-LSMT/src/cnn_lstm_project/data_plots/optimized_dataloader')
    # visualize_label_distributions(train_dataloader, 54, '/home/talhamuh/water-research/CNN-LSMT/src/cnn_lstm_project/data_plots/min-max-optimized-dataloader')
    # exit()
    # Access individual variables
    # ppt = data['ppt']
    # tmin = data['tmin']
    # tmax = data['tmax']

    # labels = pd.read_csv(config['labels_path'])
    # labels = labels.iloc[:1825]
    # min_length = min(len(ppt), len(tmin), len(tmax), len(labels))
    
    # labels_train = pd.read_csv(config['labels_path']+'_train_54.csv')
    # labels_val = pd.read_csv(config['labels_path']+'_val_10.csv')
    # labels_train = labels_train.iloc[:1825]
    # labels_val = labels_val.iloc[:1825]

    # Ensure consistency between input data and labels
    # min_length = min(len(ppt), len(tmin), len(tmax), len(labels_train))
    # min_length = 1825
    # ppt = ppt[:min_length]
    # tmin = tmin[:min_length]
    # tmax = tmax[:min_length]
    # labels = labels[:min_length]
    # train_labels = labels_train[:min_length]
    # val_labels = labels_val[:min_length]
    
    
    # Prepare datasets
    # Split the labels into training and valing
    # train_labels = labels.iloc[:, 1:45]  # First 45 labels for training
    # val_labels = labels.iloc[:, 45:55]   # Last 19 labels for valing
    # test_labels = labels.iloc[:, 55:]   # Last 19 labels for valing
    # print(train_labels.shape, val_labels.shape, test_labels.shape)
    # # exit()
    # # Create separate datasets for training and valing
    # train_dataset = ClimateDataset(ppt, tmin, tmax, train_labels)
    # val_dataset = ClimateDataset(ppt, tmin, tmax, val_labels)
    # test_dataset = ClimateDataset(ppt, tmin, tmax, test_labels)
    # # Now split the dataset
    # print(train_dataset.labels)
    # print("Train PPT:", train_dataset.ppt.shape)
    # print("Train TMIN:",train_dataset.tmin.shape)
    # print("Train TMAX:",train_dataset.tmax.shape)
    # print("Train labels:",train_dataset.labels.shape)
    # print(val_dataset.labels)
    # print("val PPT:", val_dataset.ppt.shape)
    # print("val TMIN:",val_dataset.tmin.shape)
    # print("val TMAX:",val_dataset.tmax.shape)
    # print("val labels:",val_dataset.labels.shape)
    # print(test_dataset.labels)
    # print("val PPT:", test_dataset.ppt.shape)
    # print("val TMIN:",test_dataset.tmin.shape)
    # print("val TMAX:",test_dataset.tmax.shape)
    # print("val labels:",test_dataset.labels.shape)
    # exit()
    # Split based on predefined indices
    # train_indices = list(range(0, int(len(dataset) * 0.8)))  # First 80% for training
    # val_indices = list(range(int(len(dataset) * 0.8), len(dataset)))  # Remaining 20% for valing
    
    # train_dataset = torch.utils.data.Subset(dataset, train_indices)
    # val_dataset = torch.utils.data.Subset(dataset, val_indices)

    # # Prepare data loaders
    # train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=32)
    # val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=32)
    # train_loader = DataLoader(dataset, batch_size=config['batch_size'])
    # visualize_top_examples(train_loader, '/home/talhamuh/water-research/CNN-LSMT/src/cnn_lstm_project/data_plots/first_100_pixel_norm', 365)
    # exit()
    # Initialize model, optimizer, and loss function
    model = CNN_LSTM().to(device)
    start_epoch = 0
    model = nn.DataParallel(model, device_ids=[1, 2])  # Multi-GPU support with DataParallel
    # Freeze the CNN and LSTM layers
    # model.freeze_backbone()

    # Replace the last fully connected layer if needed
    # num_features = model.fc3.in_features  # Get input size of the last FC layer
    # model.fc3 = nn.Linear(num_features, 44).to(device)  # Change output layer for 44 locations
    # print(model)
    # Only the new fully connected layer's parameters will be updated during training
    # optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config['lr'])
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    criterion = nn.MSELoss()

    if config['resume']:
        model, optimizer, scheduler, start_epoch = load_checkpoint(config['checkpoint_path'], model, optimizer, device)

    # if config['mode'] == 'infer':
    #     model, optimizer, scheduler, start_epoch = load_checkpoint(config['checkpoint_path'], model, optimizer, device)
    #     # Define batch size and input dimensions
    #     # batch_size = 1  # Adjust as needed
    #     # height, width = 1849, 1458  # Replace with your actual spatial dimensions

    #     # Create dummy inputs for ppt, tmin, tmax
    #     # ppt_dummy = torch.randn(batch_size, height, width)
    #     # tmin_dummy = torch.randn(batch_size, height, width)
    #     # tmax_dummy = torch.randn(batch_size, height, width)
    #     # writer.add_graph(model, (ppt_dummy, tmin_dummy, tmax_dummy))
    #     inference_loader = DataLoader(test_dataset, batch_size=config['batch_size'])
    #     inference(model, inference_loader, device, 'results/')
    elif config['mode'] == 'train':
        for epoch in range(start_epoch, config['epochs']):
            train_loss, train_nse, train_mse = train_model(model, train_loader, optimizer, criterion, device, writer, epoch)
            writer.add_scalar('Loss/Train', train_loss, epoch)
            writer.add_scalar('NSE/Train', train_nse, epoch)
            writer.add_scalar('MSE/Train', train_mse, epoch)
            print(f"Epoch [{epoch+1}/{config['epochs']}], Train Loss: {train_loss:.4f}, Train NSE: {train_nse:.4f}, Train MSE: {train_mse:.4f}")
            
            if (epoch + 1) % 25 == 0:
                # save_checkpoint({
                #     'epoch': epoch,
                #     'state_dict': model.state_dict(),
                #     'optimizer': optimizer.state_dict(),
                # }, filename=config['save_checkpoint_path'])
                # Save model without DataParallel wrapper
                save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, filename=config['save_checkpoint_path'])

                val_loss, val_nse, val_mse = val_model(model, val_loader, criterion, device, writer, epoch)
                writer.add_scalar('Loss/val', val_loss, epoch)
                writer.add_scalar('NSE/val', val_nse, epoch)
                writer.add_scalar('MSE/val', val_mse, epoch)
                print(f"Epoch [{epoch+1}/{config['epochs']}], val Loss: {val_loss:.4f}, val NSE: {val_nse:.4f}, Train MSE: {val_mse:.4f}")

    writer.close()

if __name__ == "__main__":
    main()
