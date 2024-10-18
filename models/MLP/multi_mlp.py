import argparse
from torch import nn
import torch

class Multi_MLP(nn.Module):
    def __init__(self, node_num, input_size, hidden_sizes, output_size):
        super(Multi_MLP, self).__init__()
        self.node_num = node_num
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.model_dict = nn.ModuleDict()
        for i in range(node_num):
            layers = []
            prev_size = input_size
            for hidden_size in hidden_sizes:
                layers.append(nn.Linear(prev_size, hidden_size))
                layers.append(nn.ReLU())
                prev_size = hidden_size
            layers.append(nn.Linear(prev_size, output_size))
            self.model_dict.add_module(f'{i}_layer', nn.Sequential(*layers))

    def forward(self, x):
        # [Batch, Input_len, Node]
        for i in range(self.node_num):
            if i==0:
                y = self.model_dict[f'{i}_layer'](x[:, :, i])
                y = y.unsqueeze(-1)
            else:
                y = torch.concatenate([y, self.model_dict[f'{i}_layer'](x[:, :, i]).unsqueeze(-1)], dim=2)

        return y

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--window_size', type=int, default=12)
    parser.add_argument('--horizon', type=int, default=3)
    parser.add_argument('--hidden_sizes', type=list, default=[36])

    args = parser.parse_args()

    model = Multi_MLP(node_num=96, input_size=args.window_size, hidden_sizes=args.hidden_sizes, output_size=args.horizon)

    print(model)

    x = torch.randn(8, 12, 96)
    y = model(x)
    print(y.shape)
