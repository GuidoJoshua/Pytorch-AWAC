import torch.nn as nn
import torch

class MLP(nn.Module):

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 num_neurons: list = [64, 32],
                 hidden_act: str = 'ReLU',
                 out_act: str = 'Identity'):
        super(MLP, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_neurons = num_neurons
        self.hidden_act = getattr(nn, hidden_act)()
        self.out_act = getattr(nn, out_act)()

        input_dims = [input_dim] + num_neurons
        output_dims = num_neurons + [output_dim]

        self.layers = nn.ModuleList()
        for i, (in_dim, out_dim) in enumerate(zip(input_dims, output_dims)):
            is_last = True if i == len(input_dims) - 1 else False
            self.layers.append(nn.Linear(in_dim, out_dim))
            if is_last:
                self.layers.append(self.out_act)
            else:
                self.layers.append(self.hidden_act)

    def forward(self, xs):
        for layer in self.layers:
            xs = layer(xs)
        return xs

class EnsembleNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, num_neurons, out_act="ReLU", num_critics=5, lam=0.9):
        super(EnsembleNetwork, self).__init__()
        self.nets = nn.ModuleList([
            MLP(state_dim, action_dim, num_neurons) for _ in range(num_critics)
        ])
        self.lam = lam
    
    def min_value(self, state):
        values_list = [mlp(state) for mlp in self.nets]
        values_list = torch.stack(values_list, dim=0)
        return torch.min(values_list, dim=0).values
    
    def max_value(self, state):
        values_list = [mlp(state) for mlp in self.nets]
        values_list = torch.stack(values_list, dim=0)
        return torch.max(values_list, dim=0).values
    
    def forward(self, state):
        min_value = self.min_value(state)
        max_value = self.max_value(state)
        return self.lam * min_value + (1 - self.lam) * max_value
    

class FNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes=(256, 256)):
        super(FNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], 1)  # Output a scalar value
        )

    def forward(self, state, action):
        # Concatenate state and action along the feature dimension
        x = torch.cat([state, action], dim=-1)
        return self.net(x)
