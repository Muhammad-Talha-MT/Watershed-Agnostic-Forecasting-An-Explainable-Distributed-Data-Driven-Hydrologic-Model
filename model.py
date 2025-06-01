import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
from torchvision.models import resnet18, resnet101, ResNet18_Weights, ResNet101_Weights

class CNN_LSTM(nn.Module):
    def __init__(self):
        super(CNN_LSTM, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.batch_norm1 = nn.BatchNorm2d(16)
        self.batch_norm2 = nn.BatchNorm2d(32)
        self.batch_norm3 = nn.BatchNorm2d(64)
        self.dropout_cnn = nn.Dropout(0.1)
        
        # Calculate CNN output dimensions
        cnn_output_height = 1849 // 8  # Adjust based on pooling and strides
        cnn_output_width = 1458 // 8
        cnn_output_channels = 64
        cnn_output_size = cnn_output_channels * cnn_output_height * cnn_output_width
        
        # LSTM layers
        self.lstm = nn.LSTM(167040, 512, num_layers=2, batch_first=True, dropout=0.1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 61)
        
        # Initialize biases
        nn.init.constant_(self.fc1.bias, 0.1)
        nn.init.constant_(self.fc2.bias, 0.1)
        nn.init.constant_(self.fc3.bias, 0.1)
        
        # ReLU activation
        self.relu = nn.ReLU()

    def forward(self, ppt, tmin, tmax):
        # Concatenate input tensors along the channel dimension
        x = torch.cat((ppt.unsqueeze(2), tmin.unsqueeze(2), tmax.unsqueeze(2)), dim=2)
        batch_size, seq_len, _, height, width = x.shape
        
        # Process each timestep independently through the CNN
        cnn_features = []
        for t in range(seq_len):
            x_t = x[:, t]  # Extract the t-th timestep: [batch_size, 3, height, width]
            x_t = self.pool(self.batch_norm1(F.relu(self.conv1(x_t))))
            x_t = self.pool(self.batch_norm2(F.relu(self.conv2(x_t))))
            x_t = self.pool(self.batch_norm3(F.relu(self.conv3(x_t))))
            x_t = x_t.view(batch_size, -1)  # Flatten for LSTM input
            cnn_features.append(x_t)
        
        # Stack CNN outputs along the temporal dimension for LSTM input
        x = torch.stack(cnn_features, dim=1)  # [batch_size, seq_len, cnn_output_size]

        # Apply LSTM layers
        x, _ = self.lstm(x)
        
        # Use the output of the 5th day (last time step)
        x = x[:, -1, :]  # [batch_size, 512]
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # Final output: [batch_size, 61]
        
        return x

    def freeze_backbone(self):
        # Freeze CNN and LSTM layers
        for name, param in self.named_parameters():
            if not name.startswith('fc'):
                param.requires_grad = False


class ResNet_LSTM(nn.Module):
    def __init__(self):
        super(ResNet_LSTM, self).__init__()

        # Load a pretrained ResNet
        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)

        # Modify the ResNet to remove the final fully connected layer
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()  # Removing the fully connected layer

        # Hardcoded output size based on ResNet18's final layer output before the FC layer
        self.resnet_output_size = 512  # Assuming this is correct for your input size

        # LSTM layers
        self.lstm = nn.LSTM(input_size=self.resnet_output_size, hidden_size=256, num_layers=2, batch_first=True, dropout=0.5)

        # Fully connected layers
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

        # Dropout layers
        self.dropout_lstm = nn.Dropout(0.5)
        # Initialize weights
        self.apply(self.initialize_weights)

    def initialize_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
                
    def forward(self, ppt, tmin, tmax):
        # print(f"ppt shape: {ppt.shape}")
        # print(f"tmin shape: {tmin.shape}")
        # print(f"tmax shape: {tmax.shape}")
        # Concatenate the input tensors along the channel dimension
        x = torch.cat((ppt.unsqueeze(1), tmin.unsqueeze(1), tmax.unsqueeze(1)), dim=1)
        # print(f"Concatenated tensor shape: {x.shape}")
        # Pass through ResNet
        x = self.resnet(x)

        # Flatten the output for LSTM input
        x = x.view(x.size(0), -1)
        x = x.unsqueeze(1)  # Adding sequence dimension

        # Pass through LSTM
        x, _ = self.lstm(x)

        # Apply dropout after LSTM layers
        x = self.dropout_lstm(x)

        # Use the output from the last LSTM time step
        x = x[:, -1, :]
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))

        # Apply ReLU activation
        # x = self.relu(x)

        return x
    
class PretrainedCNNLSTM(nn.Module):
    """
    Fast CNN-LSTM using a pretrained ResNet18 backbone and GRU.
    - Inputs: ppt, tmin, tmax as [B, T, H, W]
    - Backbone: ResNet18 pretrained, truncated before avgpool
    """
    def __init__(self, num_outputs=61, freeze_backbone=True):
        super(PretrainedCNNLSTM, self).__init__()
        # Load ResNet18 backbone with latest default weights
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        # Remove fully connected layer and avgpool
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # outputs [B, 512, H', W']
        # Global pooling to collapse spatial dims -> [B, 512]
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        feature_dim = 512

        # Optionally freeze backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Temporal GRU instead of LSTM for speed
        self.gru = nn.GRU(feature_dim, 128, batch_first=True, dropout=0.0)
        # Final classification head
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_outputs)
        )

    def forward(self, ppt, tmin, tmax):
        # Stack inputs along channel dim: [B, T, 3, H, W]
        x = torch.stack((ppt, tmin, tmax), dim=2)
        B, T, C, H, W = x.shape

        # Merge batch and time for single CNN pass
        x = x.view(B * T, C, H, W)
        # Backbone feature maps: [B*T, 512, H', W']
        feat = self.backbone(x)
        # Global pooling and reshape back to sequence: [B, T, 512]
        feat = self.global_pool(feat).view(B, T, -1)

        # Temporal modeling
        out, _ = self.gru(feat)         # [B, T, 128]
        last = out[:, -1, :]            # [B, 128]

        # Classification
        return self.classifier(last)

    def freeze_backbone(self):
        """Unfreeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True

