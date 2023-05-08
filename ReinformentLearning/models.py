import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.utils import degree
from torch_geometric.nn import GCNConv, SAGEConv

class Actor(nn.Module):
    def __init__(self):
        """ActorCritic algortihm.
        Args:
            state_space (int): input size of state.
            action_space (int): output size of action for policy.
        """
        super().__init__()
        self.conv1 = GCNConv(1, 256)
        self.conv2 = SAGEConv(256, 1)
        self.policy1 = nn.Linear(3, 64)
        self.policy2 = nn.Linear(64, 5)
        
        
    def forward(self, state):
        """Forward propagation.
        Args:
            state (PyG PairData): state representation.
        """
        x = state.x
        edge_index = state.edge_index
        # actor networks
        z1 = self.conv1(x, edge_index)
        a1 = F.relu(z1)
        a2 = self.conv2(a1, edge_index)

        deg = degree(state.edge_index[0], state.num_nodes)
        deg = torch.reshape(deg, (-1, 1))
        x = torch.cat((a2, deg, state.x), dim=-1)
        # Softmax output for each action and its log probabilities
        logit = self.policy1(x)
        logit = F.relu(logit)
        
        logit = self.policy2(logit)
        logit = logit.reshape((-1,) + logit.shape[2:])
        proba = nn.Softmax(dim=-1)(logit)
        log_proba = nn.LogSoftmax(dim=-1)(logit)

        return proba, log_proba

class Critic(nn.Module):

    def __init__(self):
        """ActorCritic algortihm.
        Args:
            state_space (int): input size of state.
            action_space (int): output size of action for policy.
        """
        super().__init__()
        self.conv1 = GCNConv(2, 128)
        self.conv2 = SAGEConv(128, 128)
        self.value1 = nn.Linear(128, 64)
        self.value2 = nn.Linear(64, 1)
        
    def forward(self, state):
        """Forward propagation.
        Args:
            state (PyG PairData): state representation.
        """
        x = state.x
        deg = degree(state.edge_index[0], state.num_nodes)
        deg = torch.reshape(deg, (-1, 1))
        x = torch.cat((state.x, deg), dim=-1)
        edge_index = state.edge_index
        z1 = self.conv1(x, edge_index)
        a1 = F.relu(z1)
        a2 = self.conv2(a1, edge_index)
               

        # Value function
        value = self.value1(a2)
        value = F.relu(value)
        value = self.value2(value)
        value = torch.sum(value)

        return value