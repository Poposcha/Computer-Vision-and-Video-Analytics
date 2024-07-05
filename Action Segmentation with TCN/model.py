import torch
import torch.nn as nn

class SingleStageTCN(nn.Module):
    def __init__(self, num_layers, num_filters, kernel_size, num_classes, in_channels=2048):
        super(SingleStageTCN, self).__init__()
        self.layers = nn.ModuleList()
        self.num_layers = num_layers
        self.num_filters = num_filters
        self.in_channels = in_channels
        
        # Initial 1x1 convolution to adjust the dimension of input features
        self.first_layer = nn.Conv1d(in_channels=in_channels, out_channels=num_filters, kernel_size=1)
        
        # Dilated convolution layers
        dilation = 0
        for i in range(num_layers):
            dilation += 1
            self.layers.append(nn.Conv1d(in_channels=num_filters, out_channels=num_filters, kernel_size=kernel_size, dilation=dilation, padding=dilation))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Conv1d(in_channels=num_filters, out_channels=num_filters, kernel_size=1))
        
        # Final 1x1 convolution to produce class scores
        self.last_layer = nn.Conv1d(in_channels=num_filters, out_channels=num_classes, kernel_size=1)
        
    def forward(self, x):
        out = x


        out = self.first_layer(x)
        for i in range(self.num_layers):
            residual = out
            out = self.layers[i*3](out)
            out = self.layers[i*3+1](out)
            out = self.layers[i*3+2](out)
            out = out + residual

        
        out = self.last_layer(out)

        return out
    
class MultiStageTCN(nn.Module):
    def __init__(self, num_stages, num_layers, num_filters, kernel_size, num_classes):
        super(MultiStageTCN, self).__init__()
        self.stages = nn.ModuleList([SingleStageTCN(num_layers, num_filters, kernel_size, num_classes, in_channels=2048 if  _ ==0 else num_classes) for _ in range(num_stages)])
        self.num_classes = num_classes
        
    def forward(self, x):
        out = x

        for stage in self.stages:
            out = stage(out)
        return out

    