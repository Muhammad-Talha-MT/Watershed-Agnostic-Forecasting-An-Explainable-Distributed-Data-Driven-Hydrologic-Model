import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns

# Function to unnormalize the data for visualization if necessary
def unnormalize(tensor):
    # Assume that data was normalized between [0, 1]. Adjust as per your normalization.
    return tensor * 255.0

def visualize_top_examples(dataloader, save_dir, examples_per_batch=5):
    os.makedirs(save_dir, exist_ok=True)
    
    for i, data in enumerate(dataloader):
        ppt = data['ppt'].cpu().numpy()
        tmin = data['tmin'].cpu().numpy()
        tmax = data['tmax'].cpu().numpy()
        
        # Plot and save the first 5 examples from the batch
        num_samples_in_batch = min(examples_per_batch, ppt.shape[0])  # Plot up to 5 examples per batch
        
        for j in range(num_samples_in_batch):
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            
            # PPT plot with color bar
            im0 = axs[0].imshow(ppt[j], cmap='Blues', aspect='auto')
            axs[0].set_title(f'PPT Example {i}')
            fig.colorbar(im0, ax=axs[0], orientation='vertical')
            
            # TMIN plot with color bar
            im1 = axs[1].imshow(tmin[j], cmap='Reds', aspect='auto')
            axs[1].set_title(f'TMIN Example {i}')
            fig.colorbar(im1, ax=axs[1], orientation='vertical')
            
            # TMAX plot with color bar
            im2 = axs[2].imshow(tmax[j], cmap='Reds', aspect='auto')
            axs[2].set_title(f'TMAX Example {i}')
            fig.colorbar(im2, ax=axs[2], orientation='vertical')
            
            # Save the figure
            plt.savefig(os.path.join(save_dir, f'sample_{i}.png'))
            plt.close(fig)




def visualize_box_plot(dataloader, save_dir, num_days=15):
    os.makedirs(save_dir, exist_ok=True)
    
    for i, data in enumerate(dataloader):
        if i >= 1:  # We only need one figure for the first batch
            break
        
        ppt = data['ppt'].cpu().numpy()
        tmin = data['tmin'].cpu().numpy()
        tmax = data['tmax'].cpu().numpy()
        
        fig, axs = plt.subplots(3, num_days, figsize=(num_days * 4, 15))
        
        # Determine global y-axis limits for each variable
        ppt_min, ppt_max = ppt[:num_days].min(), ppt[:num_days].max()
        tmin_min, tmin_max = tmin[:num_days].min(), tmin[:num_days].max()
        tmax_min, tmax_max = tmax[:num_days].min(), tmax[:num_days].max()
        
        for day in range(num_days):
            # Flatten the 2D frame to 1D for box plot
            ppt_day = ppt[day].flatten()
            tmin_day = tmin[day].flatten()
            tmax_day = tmax[day].flatten()
            
            axs[0, day].boxplot(ppt_day, patch_artist=True)
            axs[0, day].set_title(f'PPT - Day {day+1}')
            axs[0, day].set_ylim([ppt_min, ppt_max])
            axs[0, day].set_ylabel('PPT')
            
            axs[1, day].boxplot(tmin_day, patch_artist=True)
            axs[1, day].set_title(f'TMIN - Day {day+1}')
            axs[1, day].set_ylim([tmin_min, tmin_max])
            axs[1, day].set_ylabel('TMIN')
            
            axs[2, day].boxplot(tmax_day, patch_artist=True)
            axs[2, day].set_title(f'TMAX - Day {day+1}')
            axs[2, day].set_ylim([tmax_min, tmax_max])
            axs[2, day].set_ylabel('TMAX')
        
        # Adjust layout to prevent overlap
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(os.path.join(save_dir, 'box_plot_all_days_z_scale.png'))
        plt.close(fig)