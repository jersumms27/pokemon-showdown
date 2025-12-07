import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class DQN(nn.Module):
    def __init__(self, state_size: int,
                 num_actions: int,
                 hidden_sizes: list[int] = [],
                 dropout: float = 0.0
    ) -> None:
        super().__init__()

        if not hidden_sizes:
            hidden_sizes = [state_size]
        
        self.input_layer: nn.Linear = nn.Linear(state_size, hidden_sizes[0])
        self.hidden_layers: nn.ModuleList = nn.ModuleList([nn.Linear(hidden_sizes[i], hidden_sizes[i+1]) for i in range(len(hidden_sizes)-1)])
        self.output_layer: nn.Linear = nn.Linear(hidden_sizes[-1], num_actions)

        self.dropout: nn.Dropout = nn.Dropout(dropout)


    def forward(self, state: Tensor) -> Tensor:
        output: Tensor = F.relu(self.input_layer(state))

        for hidden_layer in self.hidden_layers:
            residual: Tensor = output
            output = F.relu(hidden_layer(output))
            
            if residual.shape == output.shape:
                output += residual

            output = self.dropout(output)
        
        q_values: Tensor = self.output_layer(output)
        return q_values