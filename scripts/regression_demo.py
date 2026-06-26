import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from neuronseek.models.proxy_model import ProxyModel, STRidge


def regression_experiment():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def ground_truth(x):
        return 2.5 * x[:, 0] + 3.0 * (x[:, 1] ** 2) - 1.2 * (x[:, 2] ** 3)
    
    torch.manual_seed(42)
    X = torch.randn(1000, 5)
    y = ground_truth(X).unsqueeze(1)
    
    dataset = torch.utils.data.TensorDataset(X, y)
    
    model = ProxyModel(in_dim=5, rank=3, max_order=3, task_type='regression')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    stridge = STRidge(model, optimizer, l1_lambda=1e-3)
    
    model.to(device)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    criterion = torch.nn.MSELoss()
    
    for epoch in range(100):
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            loss = stridge.train_step(x_batch, y_batch, criterion)
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}: Loss {loss:.6f}")
    
    active_orders = stridge.threshold_prune(percentile=10)
    print(f"\nActive Orders: {[o+1 for o in active_orders]}")


if __name__ == "__main__":
    regression_experiment()
