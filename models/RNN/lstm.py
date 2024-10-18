import argparse
from torch import nn
import torch

class LSTM(nn.Module):
    def __init__(self, node_num, input_len, input_channel, hidden_sizes, output_len, output_channel):
        super(LSTM, self).__init__()
        self.node_num = node_num
        self.input_len = input_len
        self.input_channel = input_channel
        self.hidden_sizes = hidden_sizes
        self.output_len = output_len
        self.layers = nn.ModuleList()
        self.output_channel = output_channel
        prev_size = node_num * input_channel
        for hidden_size in hidden_sizes:
            self.layers.append(nn.LSTM(input_size=prev_size, hidden_size=hidden_size))
            prev_size = hidden_size
        self.layers.append(nn.LSTM(input_size=prev_size, hidden_size=node_num * output_channel))
        self.mlp = nn.Linear(input_len, output_len)
    def forward(self, x):
        # [Batch, Input_len, Node, Input_channel] --> [Batch, Input_len, Node*Input_channel]
        x = x.reshape(x.shape[0], x.shape[1], -1)
        # [Batch, Input_len, Node*Input_channel] --> [Input_len, Batch, Node*Input_channel]
        x = x.permute(1, 0, 2)
        prev_output = x
        for layer in self.layers:
            output, (ht, ct) = layer(prev_output)
            prev_output = output
        # [Input_len, Batch, Node * Output_channel] --> [Batch, Node * Output_channel, Input_len]
        y = output.permute(1, 2, 0)
        # [Batch, Node * Output_channel, Input_len] --> [Batch, Node * Output_channel, Output_len]
        y = self.mlp(y)
        # [Batch, Node * Output_channel, Output_len] --> [Batch, Output_len, Node * Output_channel]
        y = y.permute(0, 2, 1)
        y = y.reshape(y.shape[0], y.shape[1], -1, self.output_channel)
        return y

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--window_size', type=int, default=12)
    parser.add_argument('--horizon', type=int, default=3)
    parser.add_argument('--hidden_sizes', type=list, default=[64, 36, 64])
    args = parser.parse_args()
    model = LSTM(node_num=96, input_len=args.window_size, input_channel=1, hidden_sizes=args.hidden_sizes,
                 output_len=args.horizon, output_channel=1)
    print(model)
    x = torch.randn(8, 12, 96, 1)
    y = model(x)
    print(y.shape)