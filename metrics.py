import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def calculate_nse(y_true, y_pred):
    mean_observed = torch.mean(y_true, dim=0)
    numerator = torch.sum((y_true - y_pred) ** 2, dim=0)
    denominator = torch.sum((y_true - mean_observed) ** 2, dim=0)
    nse = 1 - (numerator / denominator)
    return nse

def plot_metrics(model, dataloader, device, save_dir):
    """Plot and save the evaluation metrics."""
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

    
    nse = calculate_nse(torch.tensor(labels), torch.tensor(outputs))
    mse = mean_squared_error(labels, outputs)
    plt.figure(figsize=(10, 6))
    # sns.regplot(x=labels, y=outputs, scatter_kws={'alpha':0.5}, line_kws={"color":"r","lw":2})
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    # plt.title(f'Output: MSE = {mse:.4f}, NSE = {nse:.4f}')
    plt.savefig(os.path.join(save_dir, f'output.png'))
    plt.close()
    # for i in range(outputs.shape[1]):
    #     output = outputs[:, i]
    #     label = labels[:, i]
        
    #     mse = mean_squared_error(label, output)
    #     nse = calculate_nse(torch.tensor(label), torch.tensor(output)).item()

    #     mse_values.append(mse)
    #     nse_values.append(nse)

    #     r2 = r2_score(label, output)

    #     plt.figure(figsize=(10, 6))
    #     sns.regplot(x=label, y=output, scatter_kws={'alpha':0.5}, line_kws={"color":"r","lw":2})
    #     plt.xlabel('True Values')
    #     plt.ylabel('Predictions')
    #     plt.title(f'Output {i+1}: MSE = {mse:.4f}, R² = {r2:.4f}, NSE = {nse:.4f}')
    #     plt.savefig(os.path.join(save_dir, f'output_{i+1}.png'))
    #     plt.close()

    # # Calculate and save the average MSE and NSE for all 64 outputs
    # avg_mse = np.mean(mse_values)
    # avg_nse = np.mean(nse_values)

    # with open(os.path.join(save_dir, 'average_metrics.txt'), 'w') as f:
    #     f.write(f'Average MSE: {avg_mse:.4f}\n')
    #     f.write(f'Average NSE: {avg_nse:.4f}\n')

    print("Plots and metrics saved.")

# Function to evaluate the model and plot predictions vs actual values
def plot_predictions(model, dataloader, device, save_dir):
    """Plot and save the evaluation metrics."""
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
        # Denormalize the output and labels
        # output_denorm = dataset.denormalize_labels(output)
        # label_denorm = dataset.denormalize_labels(label)
        print(label.shape, label)
        print(output.shape, output)
        mse = mean_squared_error(label, output)
        nse = calculate_nse(torch.tensor(label), torch.tensor(output)).item()

        mse_values.append(mse)
        nse_values.append(nse)

        r2 = r2_score(label, output)
    

        print(f'Training Data - R²: {r2:.2f}, NSE: {nse:.2f}, MSE: {mse:.2f}')
        
        # Scatter plot for training data
        plt.figure(figsize=(10, 6))
        plt.scatter(label, output, alpha=0.6, color='blue', label='Training Data')
        plt.xlabel('Actual Streamflow')
        plt.ylabel('Predicted Streamflow')
        plt.title('Comparison of Actual and Predicted Streamflow')
        plt.plot([label.min(), label.max()], 
                [label.min(), label.max()], 'k--')  # Diagonal line
        plt.grid(True)
        plt.legend()
        
        # Add R², NSE, RMSE, and MAE to the plot for training data
        plt.text(0.05, 0.95, f'Training Data - R²: {r2:.2f}', transform=plt.gca().transAxes)
        plt.text(0.05, 0.90, f'Training Data - NSE: {nse:.2f}', transform=plt.gca().transAxes)
        plt.text(0.05, 0.85, f'Training Data - RMSE: {mse:.2f}', transform=plt.gca().transAxes)
        plt.savefig(os.path.join(save_dir, f'output_{i+1}_700.png'))
