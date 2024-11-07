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
from metrics import plot_metrics, plot_predictions
import logging

def save_checkpoint(state, filename="/home/talhamuh/water-research/CNN-LSMT/src/cnn_lstm_project/checkpoint/checkpoint_resnet_lstm_adam_1_label.pth.tar"):
    torch.save(state, filename)

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    start_epoch = checkpoint['epoch']
    
    if optimizer and scheduler:
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
    
    return model, optimizer, scheduler, start_epoch

def calculate_nse(observed, predicted):
    observed_mean = torch.mean(observed)
    print(observed_mean)
    numerator = torch.sum((predicted - observed) ** 2)
    denominator = torch.sum((observed - observed_mean) ** 2)
    nse = 1 - (numerator / denominator)
    return nse.item()


def setup_logging():
    logging.basicConfig(
        filename='/home/talhamuh/water-research/CNN-LSMT/src/cnn_lstm_project/logs/training_adam.log',
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )

def main(resume=False, checkpoint_path="/home/talhamuh/water-research/CNN-LSMT/src/cnn_lstm_project/checkpoint/checkpoint_resnet_lstm_adam_1_label.pth.tar", plot_only=False):
    # Set up logging
    setup_logging()
    
    # Set up TensorBoard writer
    writer = SummaryWriter(log_dir='/home/talhamuh/water-research/CNN-LSMT/src/cnn_lstm_project/runs/resnet_lstm_experiment_min_max_normalization_adam_1_label')
    
    h5_file = "/data/PRISM/Michigan_250m_1990_2022.h5"

    ppt = read_hdf5_data_parallel(h5_file, 'ppt', 2000, 2004)
    tmin = read_hdf5_data_parallel(h5_file, 'tmin', 2000, 2004)
    tmax = read_hdf5_data_parallel(h5_file, 'tmax', 2000, 2004)

    labels = pd.read_csv('/home/talhamuh/water-research/CNN-LSMT/data/processed/streamflow_data/combined_streamflow_all_vpuids.csv')
    labels = labels.iloc[:1825]
    logging.info('Dataset loaded')
    # Ensure consistency between input data and labels
    min_length = min(len(ppt), len(tmin), len(tmax), len(labels))
    min_length = 1825
    ppt = ppt[:min_length]
    tmin = tmin[:min_length]
    tmax = tmax[:min_length]
    labels = labels[:min_length]
    logging.info(f'Dataset loaded with {min_length} samples.')

    # Normalize and prepare dataset
    dataset = ClimateDataset(ppt, tmin, tmax, labels)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
    # visualize_box_plot(dataloader, save_dir='/home/talhamuh/water-research/CNN-LSMT/src/cnn_lstm_project/samples')
    model = ResNet_LSTM()  # or 'resnet50' for deeper ResNet

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



    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    start_epoch = 0
    if resume or plot_only:
        model, optimizer, scheduler, start_epoch = load_checkpoint(checkpoint_path, model, optimizer)
        logging.info(f"Resuming training from epoch {start_epoch + 1}")

    if plot_only:
        plot_predictions(model, dataloader, device, save_dir='/home/talhamuh/water-research/CNN-LSMT/src/cnn_lstm_project/plots/min-max')
        return


    num_epochs = 200
    accumulation_steps = 1  # Number of steps to accumulate gradients before updating

    for epoch in range(start_epoch, num_epochs):
        running_loss = 0.0
        optimizer.zero_grad()
        all_outputs = []
        all_labels = []
        for i, data in enumerate(dataloader):
            ppt = data['ppt'].to(device)
            tmin = data['tmin'].to(device)
            tmax = data['tmax'].to(device)
            labels = data['label'].to(device)
            
            outputs = model(ppt, tmin, tmax)
            labels = labels.to(outputs.device)
            loss = criterion(outputs, labels)

            loss.backward()
            
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item()

            writer.add_scalar('Loss/train', loss.item(), epoch * len(dataloader) + i)
            # Collect outputs and labels for NSE calculation
            all_outputs.append(outputs.detach().cpu())
            all_labels.append(labels.detach().cpu())
            if i % 10 == 0:
                logging.info(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}')

            torch.cuda.empty_cache()
            
            
        # Update learning rate
        # scheduler.step()
        avg_loss = running_loss / len(dataloader)
        writer.add_scalar('Loss/avg_train', avg_loss, epoch)
        logging.info('learning rate = {:.6f}'.format(optimizer.param_groups[0]['lr']))
        writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'])
        logging.info(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')

        # Calculate and log NSE after every 10 epochs
        if (epoch + 1) % 10 == 0:
            all_outputs = torch.cat(all_outputs)
            all_labels = torch.cat(all_labels)
            print(all_outputs, all_labels.shape)
            print(all_labels, all_labels.shape)
            nse = calculate_nse(all_labels, all_outputs)
            print(nse)
            writer.add_scalar('NSE/train', nse, epoch)
            logging.info(f'Epoch [{epoch+1}], NSE: {nse:.4f}')
        first_param_name, first_param_tensor = next(iter(model.named_parameters()))
        print(f"First parameter after loading checkpoint: {first_param_name}, {first_param_tensor[0,0]}")           
            
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            # 'scheduler': scheduler.state_dict(),
        })

    plot_predictions(model, dataloader, device, save_dir='/home/talhamuh/water-research/CNN-LSMT/src/cnn_lstm_project/plots/min-max')

    logging.info('Training finished.')
    writer.close()

if __name__ == "__main__":
    main(resume=True, plot_only=False)
