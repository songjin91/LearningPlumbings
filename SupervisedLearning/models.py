import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing, BatchNorm, GATConv, MLP, GCNConv

class GEmbedLayer(MessagePassing):
    def __init__(self, in_chs, hid_chs, out_chs):
        super().__init__(aggr='add')  # "Add" aggregation.
        self.in_chs = in_chs
        self.hid_chs = hid_chs
        self.out_chs = out_chs
        self._build_model()
        self.batch_norm = BatchNorm(self.out_chs)
        
    def _build_model(self):
        layer = []
        layer.append(nn.Linear(self.in_chs, self.out_chs))
        layer.append(nn.LeakyReLU())
        self.mlp_node = nn.Sequential(*layer)
        
        layer = []
        layer.append(nn.Linear(2 * self.out_chs, self.hid_chs))
        layer.append(nn.LeakyReLU())
        layer.append(nn.Linear(self.hid_chs, self.out_chs))
        self.mlp_msg = nn.Sequential(*layer)
        
        layer = []
        layer.append(nn.Linear(self.in_chs + self.out_chs, self.out_chs))
        layer.append(nn.LeakyReLU())
        self.mlp_upd = nn.Sequential(*layer)
       
    def forward(self, x, edge_index):
        x_encoded = self.mlp_node(x)
        return self.propagate(edge_index, x = x_encoded, x_original = x)

    def message(self, x_i, x_j):
        tmp = torch.cat([x_i, x_j], dim = 1)
        
        return self.mlp_msg(tmp)
    
    def update(self, aggr_out, x, x_original):
        temp = torch.cat((x_original, aggr_out), dim = 1)
        aggr_out = self.mlp_upd(temp)
        aggr_out = self.batch_norm(aggr_out)
        return aggr_out

from torch_scatter import scatter_mean

class GraphAggregator(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphAggregator, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self._build_model()
        
            
    def _build_model(self):
        layer = []
        layer.append(nn.Linear(self.in_channels, self.out_channels))
        layer.append(nn.LeakyReLU())
        self.mlp = nn.Sequential(*layer)
        
        layer = []
        layer.append(nn.Linear(self.in_channels, self.out_channels))
        self.mlp_gate = nn.Sequential(*layer)
        
        layer = []
        layer.append(nn.Linear(self.out_channels, self.hidden_channels))
        layer.append(nn.LeakyReLU())
        layer.append(nn.Linear(self.hidden_channels, self.out_channels))
        self.mlp_final = nn.Sequential(*layer)        
            
    def forward(self, x):
        x_states = self.mlp(x)
        x_gates = F.softmax(self.mlp_gate(x), dim = 1)
        x_states = x_states * x_gates
        x_states = self.mlp_final(x_states)

        return x_states

class GENGAT(torch.nn.Module):
    def __init__(self):
        super(GENGAT, self).__init__()
        self.gembed = GEmbedLayer(in_chs = 1, hid_chs = 32, out_chs = 64)
        self.gatconv = GATConv(in_channels = 64, out_channels = 64)
        
        self.aggregator = GraphAggregator(in_channels = 64, hidden_channels = 48, out_channels= 32)
        self.classification = MLP(in_channels = 64, hidden_channels = 32, out_channels = 2, num_layers = 3)

    def compute_embedding(self, x, edge_index, x_batch):        
        x = self.gembed(x, edge_index)        
        x = self.gatconv(x, edge_index)        
        x = self.aggregator(x)
        x = scatter_mean(x, x_batch, dim = 0)
        
        return x

    def forward(self, data):
        x_s, x_t = data.x_s, data.x_t
        edge_index_s, edge_index_t = data.edge_index_s, data.edge_index_t
        x_s_batch, x_t_batch = data.x_s_batch, data.x_t_batch
        embed_s = self.compute_embedding(x_s, edge_index_s, x_s_batch)
        embed_t = self.compute_embedding(x_t, edge_index_t, x_t_batch)
        out = torch.cat((embed_s, embed_t), 1)
        out = out.reshape(out.size(0), -1)
        out = self.classification(out)
        return out
    
class GCNGAT(torch.nn.Module):
    def __init__(self):
        super(GCNGAT, self).__init__()
        self.gcnconv = GCNConv(in_channels = 1, out_channels = 64)
        self.gatconv = GATConv(in_channels = 64, out_channels = 64)
        
        self.aggregator = GraphAggregator(in_channels = 64, hidden_channels = 48, out_channels= 32)
        self.classification = MLP(in_channels = 64, hidden_channels = 32, out_channels = 2, num_layers = 3)

    def compute_embedding(self, x, edge_index, x_batch):        
        x = self.gcnconv(x, edge_index)        
        x = self.gatconv(x, edge_index)        
        x = self.aggregator(x)
        x = scatter_mean(x, x_batch, dim = 0)
        
        return x

    def forward(self, data):
        x_s, x_t = data.x_s, data.x_t
        edge_index_s, edge_index_t = data.edge_index_s, data.edge_index_t
        x_s_batch, x_t_batch = data.x_s_batch, data.x_t_batch
        embed_s = self.compute_embedding(x_s, edge_index_s, x_s_batch)
        embed_t = self.compute_embedding(x_t, edge_index_t, x_t_batch)
        out = torch.cat((embed_s, embed_t), 1)
        out = out.reshape(out.size(0), -1)
        out = self.classification(out)
        return out

class GCNGCN(torch.nn.Module):
    def __init__(self):
        super(GCNGCN, self).__init__()
        self.gcnconv = GCNConv(in_channels = 1, out_channels = 64)
        self.gatconv = GCNConv(in_channels = 64, out_channels = 64)
        
        self.aggregator = GraphAggregator(in_channels = 64, hidden_channels = 48, out_channels= 32)
        self.classification = MLP(in_channels = 64, hidden_channels = 32, out_channels = 2, num_layers = 3)

    def compute_embedding(self, x, edge_index, x_batch):        
        x = self.gcnconv(x, edge_index)        
        x = self.gatconv(x, edge_index)        
        x = self.aggregator(x)
        x = scatter_mean(x, x_batch, dim = 0)
        
        return x

    def forward(self, data):
        x_s, x_t = data.x_s, data.x_t
        edge_index_s, edge_index_t = data.edge_index_s, data.edge_index_t
        x_s_batch, x_t_batch = data.x_s_batch, data.x_t_batch
        embed_s = self.compute_embedding(x_s, edge_index_s, x_s_batch)
        embed_t = self.compute_embedding(x_t, edge_index_t, x_t_batch)
        out = torch.cat((embed_s, embed_t), 1)
        out = out.reshape(out.size(0), -1)
        out = self.classification(out)
        return out