import argparse
from torch import nn
import torch

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # [Batch, Input_len, Node] --> [Batch, Node, Input_len]
        x = x.permute(0, 2, 1)
        y = self.model(x)
        # [Batch, Node, Output_len] --> [Batch, Output_len, Node]
        y = y.permute(0, 2, 1)
        return y

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--window_size', type=int, default=12)
    parser.add_argument('--horizon', type=int, default=3)
    parser.add_argument('--hidden_sizes', type=list, default=[24, 36, 24])

    args = parser.parse_args()

    model = MLP(input_size=args.window_size, hidden_sizes=args.hidden_sizes, output_size=args.horizon)

    print(model)

    x = torch.randn(8, 12, 96)
    y = model(x)
    print(y.shape)



