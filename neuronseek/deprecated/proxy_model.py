import torch
import torch.nn as nn
from neuronseek.models.cp_polynomial import CPPolynomialTerm, ClassificationHead, RegressionHead


class ProxyModel(nn.Module):
    def __init__(self, in_dim, rank, max_order, task_type, num_classes=None):
        super().__init__()
        self.task_type = task_type
        self.max_order = max_order
        
        self.terms = nn.ModuleList([
            CPPolynomialTerm(in_dim, rank, order) 
            for order in range(1, max_order + 1)
        ])
        
        if task_type == 'classification':
            assert num_classes is not None
            self.heads = nn.ModuleList([
                ClassificationHead(rank, num_classes) 
                for _ in range(max_order)
            ])
        else:
            self.heads = nn.ModuleList([
                RegressionHead(rank) 
                for _ in range(max_order)
            ])
        
        self.active_mask = nn.Parameter(
            torch.ones(max_order), requires_grad=False
        )
    
    def forward(self, x):
        outputs = []
        for term, head, mask in zip(self.terms, self.heads, self.active_mask):
            if mask > 0:
                H = term(x)
                y = head(H)
                outputs.append(y)
        
        if len(outputs) == 0:
            return 0
        return sum(outputs)
    
    def get_importance_scores(self):
        return torch.tensor([
            head.importance() * mask.item() 
            for head, mask in zip(self.heads, self.active_mask)
        ])
    
    def prune_terms(self, threshold):
        scores = self.get_importance_scores()
        self.active_mask.data = (scores > threshold).float()
        
        for i, mask in enumerate(self.active_mask):
            if mask == 0:
                for param in self.terms[i].parameters():
                    param.requires_grad = False
                for param in self.heads[i].parameters():
                    param.requires_grad = False
        
        return self.active_mask.nonzero(as_tuple=True)[0].tolist()


class STRidge:
    def __init__(self, model, optimizer, l1_lambda=1e-3):
        self.model = model
        self.optimizer = optimizer
        self.l1_lambda = l1_lambda
    
    def train_step(self, x, y, criterion):
        self.optimizer.zero_grad()
        
        pred = self.model(x)
        loss = criterion(pred, y)
        
        l1_reg = 0
        for head in self.model.heads:
            if hasattr(head, 'weight'):
                l1_reg += torch.norm(head.weight, p=1)
            else:
                l1_reg += torch.abs(head.coef)
        
        total_loss = loss + self.l1_lambda * l1_reg
        total_loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def threshold_prune(self, percentile=10):
        scores = self.model.get_importance_scores()
        threshold = torch.quantile(scores[scores > 0], percentile / 100.0)
        active_orders = self.model.prune_terms(threshold)
        return active_orders
