import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn


def get_cifar_features(batch_size=128, device='cuda', root='./data'):
    backbone = torchvision.models.resnet18(pretrained=True)
    modules = list(backbone.children())[:-1]
    feature_extractor = nn.Sequential(*modules).to(device)
    feature_extractor.eval()
    
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    testset = torchvision.datasets.CIFAR10(root=root, train=False, 
                                            download=True, transform=transform)
    loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            feats = feature_extractor(inputs)
            feats = feats.view(feats.size(0), -1)
            all_features.append(feats.cpu())
            all_labels.append(labels)
    
    X = torch.cat(all_features, dim=0)
    y = torch.cat(all_labels, dim=0)
    
    return TensorDataset(X, y)
