import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
from torchvision.models import resnet18, resnet101, ResNet18_Weights, ResNet101_Weights

class CNN_LSTM(nn.Module):
    def __init__(self):
        super(CNN_LSTM, self).__init__()
        
        # Convolutional layers with reduced feature map size
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)  # Adjusted stride
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # Adjusted stride
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.batch_norm1 = nn.BatchNorm2d(16)
        self.batch_norm2 = nn.BatchNorm2d(32)
        self.batch_norm3 = nn.BatchNorm2d(64)
        self.dropout_cnn = nn.Dropout(0.1)
        self.dropout_lstm = nn.Dropout(0.1)
        
        # Recalculate size after CNN layers
        cnn_output_height = 1849 // 32  # Adjust based on pooling and strides
        cnn_output_width = 1458 // 32  # Adjust based on pooling and strides
        cnn_output_channels = 64
        cnn_output_size = cnn_output_channels * cnn_output_height * cnn_output_width
        # LSTM layers
        self.lstm = nn.LSTM(167040, 512, num_layers=2, batch_first=True)
        
        # Fully connected layersMax
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        nn.init.constant_(self.fc1.bias, 0.1)  # Initialize bias to a small positive value
        nn.init.constant_(self.fc2.bias, 0.1)  # Initialize bias to a small positive value
        nn.init.constant_(self.fc3.bias, 0.1)  # Initialize bias to a small positive value
        # ReLU activation
        self.relu = nn.ReLU()
        # Initialize weights

    def forward(self, ppt, tmin, tmax):
        # print("Input:", ppt.shape, tmin.shape, tmax.shape)
        # Concatenate the input tensors along the channel dimension
        x = torch.cat((ppt.unsqueeze(1), tmin.unsqueeze(1), tmax.unsqueeze(1)), dim=1)  #unsqueeze(1) is to convert input into 3D
        # print("X input: ", x.shape)
        
        # Apply convolutional layers with batch normalization and pooling
        x = self.pool(self.batch_norm1(F.relu(self.conv1(x))))
        
        x = self.pool(self.batch_norm2(F.relu(self.conv2(x))))
        x = self.pool(self.batch_norm3(F.relu(self.conv3(x))))
        # Apply dropout after convolutional layers
        # x = self.dropout_cnn(x)
        # print("Size after CNN: ", x.shape)
        
        # Flatten the feature maps for LSTM input
        x = x.view(x.size(0), -1)
        # print("Size after view: ", x.shape)
        x = x.unsqueeze(1)  # Adding sequence dimension
        # print("Size after unsqueeze: ", x.shape)
        # Apply LSTM layers
        x, _ = self.lstm(x)
        # print("Size after LSTM output: ", x.shape)
        # exit()
        
        # Apply dropout after LSTM layers
        # x = self.dropout_lstm(x)
        
        # Use the output from the last LSTM time step
        x = x[:, -1, :]
        # Apply the fully connected layers
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        # print(x)
        # Apply ReLU activation
        
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