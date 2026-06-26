import torch.nn as nn
import torch.nn.functional as F
from neuronseek.core.task_neuron_layers import PolynomialConv2d


class BasicBlockTN(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, active_orders=[1]):
        super(BasicBlockTN, self).__init__()
        
        self.conv1 = PolynomialConv2d(in_planes, planes, kernel_size=3, stride=stride, 
                                       padding=1, bias=False, active_orders=active_orders)
        self.conv2 = PolynomialConv2d(planes, planes, kernel_size=3, stride=1, 
                                       padding=1, bias=False, active_orders=active_orders)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, 
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class CustomResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, active_orders=[1]):
        super(CustomResNet, self).__init__()
        self.in_planes = 64
        self.active_orders = active_orders

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.active_orders))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18_TN(active_orders=[1]):
    return CustomResNet(BasicBlockTN, [2, 2, 2, 2], active_orders=active_orders)
