import torch
import torch.nn as nn


class TaskDrivenLinear(nn.Module):
    def __init__(self, in_features, out_features, active_orders, rank=32):
        super().__init__()
        self.active_orders = sorted(active_orders)
        
        self.layers = nn.ModuleDict()
        self.norms = nn.ModuleDict()
        
        for order in active_orders:
            if order == 1:
                self.layers[str(order)] = nn.Linear(in_features, out_features)
            else:
                bottleneck_dim = rank
                self.layers[str(order)] = nn.Sequential(
                    nn.Linear(in_features, bottleneck_dim),
                    nn.Linear(bottleneck_dim, out_features)
                )
            self.norms[str(order)] = nn.BatchNorm1d(out_features)
    
    def forward(self, x):
        outputs = []
        for order in self.active_orders:
            if order == 1:
                feat = x
            else:
                feat = torch.pow(x, order)
            
            y = self.layers[str(order)](feat)
            y = self.norms[str(order)](y)
            outputs.append(y)
        
        return sum(outputs)


class TaskDrivenConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, 
                 active_orders, stride=1, padding=0, rank=32):
        super().__init__()
        self.active_orders = sorted(active_orders)
        
        self.layers = nn.ModuleDict()
        self.norms = nn.ModuleDict()
        
        for order in active_orders:
            if order == 1:
                self.layers[str(order)] = nn.Conv2d(
                    in_channels, out_channels, kernel_size, stride, padding
                )
            else:
                bottleneck = rank
                self.layers[str(order)] = nn.Sequential(
                    nn.Conv2d(in_channels, bottleneck, 1),
                    nn.Conv2d(bottleneck, out_channels, kernel_size, stride, padding)
                )
            self.norms[str(order)] = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        outputs = []
        for order in self.active_orders:
            if order == 1:
                feat = x
            else:
                feat = torch.pow(x, order)
            
            y = self.layers[str(order)](feat)
            y = self.norms[str(order)](y)
            outputs.append(y)
        
        return sum(outputs)
