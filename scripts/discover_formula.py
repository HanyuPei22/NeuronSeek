import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuronseek.utils.feature_extractor import get_cifar_features
from neuronseek.models.search_agent import SearchAgent


def main():
    dataset = get_cifar_features(batch_size=128)
    agent = SearchAgent(input_dim=512, num_classes=10, rank=32, poly_order=6)
    agent.fit_stridge(dataset, epochs=20)
    
    orders = agent.get_discovered_orders()
    print(f"\n>>> FINAL RESULT: Significant Structure Discovered <<<")
    print(f"Pure Terms (x^n): {orders['pure']}")
    print(f"Interaction Terms (CP): {orders['interact']}")


if __name__ == "__main__":
    main()
