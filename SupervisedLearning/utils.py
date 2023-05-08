# -*- coding: utf-8 -*-
import torch
import numpy as np
import networkx as nx
from torch_geometric.utils.convert import from_networkx
from torch_geometric.utils import to_dense_adj
from torch_geometric.data import Data


def create_graph(x, a):
    # Create a graph as a `Data` object implemented in PyG (Pytorch Geometric) from node features `x` and
    # adjacency matrix `a`.
    # `x` and `a` are supposed to be dense numpy arrays.
    pyg_graph = Data()
    nx_graph = nx.from_numpy_matrix(a) # create a 'networkx' graph from a dense adjacency matrix 'a'
    pyg_graph = from_networkx(nx_graph) # convert a networkx graph into a pyg graph 
    pyg_graph.weight = None
    pyg_graph.x = torch.tensor(x.reshape(-1, 1), dtype = torch.float32) # feeding node features
    #pyg_graph.y = y # feeding labels
    return pyg_graph

def neumann_move(g_graph, neuman_move_type = np.random.randint(1,4)):
    # This function returns a graph obtained by applying a Neumann move to the input graph `g_graph`.
    # `g_graph` is an instance of `Data` object.
    # `neuman_move_type` denotes the type of Neumann move to be applied. Its values are integers from 1 to 3.
    # The default value of `neuman_move_type` is a random integer from 1 to 3.

    # Step 1: Get node features and adjacency matrix as numpy dense arrays.
    node_label = g_graph.x.numpy()
    node_label = node_label.flatten()
    #graph_y = g_graph.y
    n = len(node_label)
    if n == 1:
        adjacency = np.zeros((1,1))
    else:
        adjacency = to_dense_adj(g_graph.edge_index)
        adjacency = adjacency.numpy()[0, :, :]

    # Step 2: Randomly choose a vertex to which a Neumann move will be applied, and find vertices directly connected to
    #         the chosen vertex
    chosen_vertex = np.random.randint(0, n)
    linked_vs=[]
    for j in range(n):
        if adjacency[chosen_vertex, j] == 1:
            linked_vs = np.append(linked_vs, j)
    linked_vs = np.array(linked_vs, dtype=int)

    sign = 1 if np.random.random() < 0.5 else -1 # Sign for Neumann moves of types 1 and 2

    # If a graph consists of single vertex, then the available types of Neumann moves are 2 and 3.
    if len(linked_vs) == 0:
        neuman_move_type = 2 if np.random.random() < 0.5 else 3
    
    # Step 3: Apply Neumann moves to the chosen vertex
    if neuman_move_type == 1:
        chosen_linked_vertex = linked_vs[np.random.randint(0, len(linked_vs))]
        node_label = np.append(node_label, sign)
        node_label[chosen_vertex] = node_label[chosen_vertex] + sign
        node_label[chosen_linked_vertex] = node_label[chosen_linked_vertex] + sign
        adjacency = np.append(adjacency, np.zeros((n, 1)), axis = 1)
        adjacency[chosen_vertex, n] = adjacency[chosen_linked_vertex, n ] = 1
        adjacency = np.append(adjacency, np.zeros((1, n + 1)), axis = 0)
        adjacency[n, chosen_vertex] = adjacency[n, chosen_linked_vertex] = 1
        adjacency[chosen_vertex, chosen_linked_vertex] = adjacency[chosen_linked_vertex, chosen_vertex] = 0
    elif neuman_move_type == 2:
        node_label = np.append(node_label, sign)
        node_label[chosen_vertex] = node_label[chosen_vertex] + sign
        adjacency = np.append(adjacency, np.zeros((n, 1)), axis = 1)
        adjacency[chosen_vertex, n] = 1
        adjacency = np.append(adjacency, np.zeros((1, n + 1)), axis = 0)
        adjacency[n, chosen_vertex] = 1        
    elif neuman_move_type == 3:
        node_label = np.append(node_label, 0)
        node_diff = np.random.randint(-20, 20)
        node_label = np.append(node_label, node_diff)
        node_label[chosen_vertex] = node_label[chosen_vertex] - node_diff
        adjacency = np.append(adjacency, np.zeros((n, 1)), axis = 1)
        adjacency = np.append(adjacency, np.zeros((1, n + 1)), axis = 0)
        adjacency = np.append(adjacency, np.zeros((n + 1, 1)), axis = 1)
        adjacency = np.append(adjacency, np.zeros((1, n + 2)), axis = 0)
        adjacency[chosen_vertex, n] = adjacency[n, chosen_vertex] = 1
        adjacency[n, n + 1] = adjacency[n + 1, n] = 1
        if len(linked_vs) > 0:
            np.random.shuffle(linked_vs)
            n_split = np.random.randint(0, len(linked_vs))
            if n_split != 0 or n_split != len(linked_vs)-1:
                linked_chosen_2 = linked_vs[n_split:]
                for i in linked_chosen_2:
                    adjacency[chosen_vertex, i] = adjacency[i, chosen_vertex] = 0
                    adjacency[n + 1, i] = adjacency[i, n + 1] = 1
    # Output
    node_label = node_label.reshape(-1, 1)
    return create_graph(node_label, adjacency)


def random_graph():
    # Returns a random graph (plumbing TREE) as an instance of `Data` object of PyG.
    # `n` : number of nodes, a random integer from 1 to 20
    # `x` : node features, an numpy array of random integers from -10 to 9
    # `a` : adjacency matrix
    n = np.random.randint(1, 26)
    x = np.random.randint(-10, 11, size = n)
    a=np.zeros((n, n))
    for i in range(1,n):
        j=np.random.randint(i)
        a[i,j]=a[j,i]=1
    return create_graph(x, a)


class PairData(Data):
    def __init__(self, edge_index_s=None, x_s=None, edge_index_t=None, x_t=None, y = None):
        super().__init__()
        self.edge_index_s = edge_index_s
        self.x_s = x_s
        self.edge_index_t = edge_index_t
        self.x_t = x_t
        self.y = y
        
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_s':
            return self.x_s.size(0)
        if key == 'edge_index_t':
            return self.x_t.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)


def generate_dataset(n_pairs = 50, n_moves = 11, status = 'random', g1_move = True, g2_move = True):
    data_list = []
    for _ in range(0, n_pairs):
        if status == 'same':
            pos = 1
        elif status =='diff':
            pos = -1
        else:
            pos = 1 if np.random.random() < 0.6 else -1
        g1 = random_graph()
        if g1_move == False and g2_move == False:
            pos = -1
        
        #for i in range(np.random.randint(1,n_moves)):
        #  g1 = neumann_move(g1)
        if pos == 1:        
            g2 = g1
            if g1_move:
                for _ in range(np.random.randint(1, n_moves)):
                    g1 = neumann_move(g1)
            if g2_move:
                for _ in range(np.random.randint(1, n_moves)):
                    g2 = neumann_move(g2)
        else:
            g2 = random_graph()
            if g1_move:
                for _ in range(np.random.randint(1, n_moves)):
                    g1 = neumann_move(g1)
            if g2_move:
                for _ in range(np.random.randint(1, n_moves)):
                    g2 = neumann_move(g2)
        # y = torch.tensor([pos])
        if pos == 1:
            y = torch.tensor([pos])
        else:
            y = torch.tensor([0])
      
        # Save
        data_list.append(PairData(edge_index_s = g1.edge_index, x_s = g1.x, 
                                edge_index_t = g2.edge_index, x_t = g2.x, y = y))
    return data_list

def tweak_graph(g_graph):
    node_label = g_graph.x.numpy().flatten()
    node_idx = np.random.randint(0, len(node_label))
    tweaks = [-2, -1, 1, 3, 2, 1, -1, 2, -3, -2, 1, -1]
    node_label[node_idx] += np.random.choice(tweaks)
    node_label = node_label.reshape(-1, 1)
    g_graph.x = torch.tensor(node_label)
    return g_graph

def generate_dataset_tweak(n_pairs = 50):
    data_list = []
    for j in range(0, n_pairs):
        g1 = random_graph()
        g2 = tweak_graph(g1)

        for i in range(np.random.randint(1, 41)):
            g1 = neumann_move(g1)
        for i in range(np.random.randint(1, 41)):
            g2 = neumann_move(g2)

        y = torch.tensor([0])
      
        # Save
        data_list.append(PairData(edge_index_s = g1.edge_index, x_s = g1.x, 
                                edge_index_t = g2.edge_index, x_t = g2.x, y = y))
    return data_list