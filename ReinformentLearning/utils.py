import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from torch_geometric.utils.convert import from_networkx, to_networkx
from torch_geometric.utils import to_dense_adj
from torch_geometric.data import Data
from torch.distributions import Categorical

def create_graph(x, a):
    # Create a graph as a `Data` object implemented in PyG (Pytorch Geometric) from node features `x` and
    # adjacency matrix `a`.
    # `x` and `a` are supposed to be dense numpy arrays.
    pyg_graph = Data()
    nx_graph = nx.from_numpy_array(a) # create a 'networkx' graph from a dense adjacency matrix 'a'
    pyg_graph = from_networkx(nx_graph) # convert a networkx graph into a pyg graph 
    pyg_graph.weight = None
    pyg_graph.num_nodes = None
    pyg_graph.x = torch.tensor(x.reshape(-1, 1), dtype = torch.float) # feeding node features
    #pyg_graph.y = y # feeding labels
    return pyg_graph

def plot_graph(pyg_graph):
    # First convert a PyG graph to a `networkx` graph, then plot the graph by using `nx.draw` function.
    # Config 
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])

    g = to_networkx(pyg_graph, to_undirected=True) #convert to networkx graph

    labeldict = {} # node labels in the plot
    for i in range(pyg_graph.num_nodes):
        labeldict[i]=pyg_graph.x[i].item()
    # Plot
    pos_nodes = nx.kamada_kawai_layout(g)
    pos_attrs = {}
    for node, coords in pos_nodes.items():
        pos_attrs[node] = (coords[0], coords[1] + 0.08)
    nx.draw(g, pos_nodes, node_color='black', node_size=30)
    nx.draw_networkx_labels(g, pos_attrs, labels=labeldict)
    plt.show()

def random_graph(n_max):
    # Returns a random graph (plumbing TREE) as an instance of `Data` object of PyG.
    # `n` : number of nodes, a random integer from 1 to 20
    # `x` : node features, an numpy array of random integers from -10 to 9
    # `a` : adjacency matrix
    # n = np.random.randint(2, n_max)
    n = n_max
    x = np.random.randint(-20, 21, size = n)
    a=np.zeros((n, n))
    for i in range(1,n):
        j=np.random.randint(i)
        a[i,j]=a[j,i]=1
    return create_graph(x, a)

def neumann_move(g_graph, node_index, type, updown, sign):
    
    done = True
    # Step 1: Get node features and adjacency matrix as numpy dense arrays.
    node_label = g_graph.x.numpy()
    node_label = node_label.flatten()
    #graph_y = g_graph.y
    n = len(node_label)
    if n == 1:
        adjacency = np.zeros((1,1))
    else:
        adjacency = to_dense_adj(g_graph.edge_index)
        adjacency = adjacency.numpy().squeeze()
    
    # Step 2:  Find vertices directly linked to the chosen vertex
    chosen_vertex = node_index
    linked_vs=[]
    for j in range(n):
        if adjacency[chosen_vertex, j] == 1:
            linked_vs = np.append(linked_vs, j)
    linked_vs = np.array(linked_vs, dtype=int)   
    
    # Step 3: Apply Neumann moves to the chosen vertex (updown = 1)
    if updown == 1:
        if type == 1:
            # If a graph consists of single vertex, then the available types of Neumann moves are 2 and 3.
            if len(linked_vs) == 0:
                done = False
            else:
                done = True
                chosen_linked_vertex = linked_vs[np.random.randint(0, len(linked_vs))]
                node_label = np.append(node_label, sign)
                node_label[chosen_vertex] += sign
                node_label[chosen_linked_vertex] += sign
                adjacency = np.append(adjacency, np.zeros((n, 1)), axis = 1)
                adjacency[chosen_vertex, n] = adjacency[chosen_linked_vertex, n ] = 1
                adjacency = np.append(adjacency, np.zeros((1, n + 1)), axis = 0)
                adjacency[n, chosen_vertex] = adjacency[n, chosen_linked_vertex] = 1
                adjacency[chosen_vertex, chosen_linked_vertex] = adjacency[chosen_linked_vertex, chosen_vertex] = 0
    
        elif type == 2:
            node_label = np.append(node_label, sign)
            node_label[chosen_vertex] += sign
            adjacency = np.append(adjacency, np.zeros((n, 1)), axis = 1)
            adjacency[chosen_vertex, n] = 1
            adjacency = np.append(adjacency, np.zeros((1, n + 1)), axis = 0)
            adjacency[n, chosen_vertex] = 1        
    
        elif type == 3:
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
            if len(linked_vs) > 1:
                np.random.shuffle(linked_vs)
                n_split = np.random.randint(1, len(linked_vs))
                linked_chosen_2 = linked_vs[n_split:]
                for i in linked_chosen_2:
                    adjacency[chosen_vertex, i] = adjacency[i, chosen_vertex] = 0
                    adjacency[n + 1, i] = adjacency[i, n + 1] = 1
            
        # Output
        node_label = node_label.reshape(-1, 1)
        return done, create_graph(node_label, adjacency)
    elif updown == -1:
        if type == 1:
            if len(linked_vs) == 2 and np.absolute(node_label[chosen_vertex]) == 1:
                for i in linked_vs:
                    node_label[i] = node_label[i] - node_label[chosen_vertex]
                node_label = np.delete(node_label, chosen_vertex)
                adjacency[linked_vs[0], linked_vs[1]] = adjacency[linked_vs[1], linked_vs[0]] = 1
                adjacency = np.delete(adjacency, chosen_vertex, 0)
                adjacency = np.delete(adjacency, chosen_vertex, 1)
                
            else:
                done = False
                return done, g_graph
        elif type == 2:
            if len(linked_vs) == 1 and np.absolute(node_label[chosen_vertex]) == 1:
                node_label[linked_vs] = node_label[linked_vs] - node_label[chosen_vertex]
                node_label = np.delete(node_label, chosen_vertex)
                adjacency = np.delete(adjacency, chosen_vertex, 0)
                adjacency = np.delete(adjacency, chosen_vertex, 1)
            else:
                done = False
                return done, g_graph
        elif type == 3:
            if len(linked_vs) == 2 and np.absolute(node_label[chosen_vertex]) == 0:
                
                new_label = node_label[linked_vs[0]]+node_label[linked_vs[1]]
                node_label = np.append(node_label, new_label)
                del_index = np.concatenate((chosen_vertex, linked_vs), axis=None)                
                node_label = np.delete(node_label, del_index)             
                linked_vs_1=[]
                for j in range(n):
                    if j != chosen_vertex:
                        if adjacency[linked_vs[0], j] == 1 or adjacency[linked_vs[1], j] == 1:
                            linked_vs_1 = np.append(linked_vs_1, j)
                linked_vs_1 = np.array(linked_vs_1, dtype=int)
                
                adjacency = np.append(adjacency, np.zeros((n, 1)), axis = 1)
                adjacency = np.append(adjacency, np.zeros((1, n + 1)), axis = 0)
                for i in linked_vs_1:
                    adjacency[n, i] = adjacency[i, n] = 1
                adjacency = np.delete(adjacency, del_index, 0)
                adjacency = np.delete(adjacency, del_index, 1)      
                                
            else:
                done = False
                return done, g_graph
        node_label = node_label.reshape(-1, 1)
        return done, create_graph(node_label, adjacency)

# Building plumbings like E8
def append_twos(graph):
    valency_one = np.random.randint(len(graph.x))    
    edge_index = graph.edge_index
    n_nodes = np.random.randint(2,6)
    if np.random.rand() > 0.5:
        xx = torch.full((n_nodes, 1), 2)
    else:
        xx = torch.full((n_nodes, 1), -2)
    x = torch.cat((graph.x, xx))
    edge_index = torch.cat((edge_index, torch.tensor([[valency_one, len(graph.x)],[len(graph.x), valency_one]])), dim = -1)
    for i in range(n_nodes-1):
        edge_index = torch.cat((edge_index, torch.tensor([[len(graph.x)+i, len(graph.x)+i+1],[len(graph.x)+i+1, len(graph.x)+i]])), dim = -1)
    
    new_graph = Data(x=x, edge_index=edge_index)
    return new_graph

def action_dict(act_index):
    if act_index == 0:
        move_type = 1
        updown = 1
        sign = 1
        return move_type, updown, sign
    elif act_index == 1:
        move_type = 1
        updown = 1
        sign = -1
        return move_type, updown, sign
    elif act_index == 4:
        move_type = 1 # doesn't matter
        updown = -1
        sign = 1 # doesn't matter
        return move_type, updown, sign
    elif act_index == 2:
        move_type = 2
        updown = 1
        sign = 1
        return move_type, updown, sign
    elif act_index == 3:
        move_type = 2
        updown = 1
        sign = -1
        return move_type, updown, sign
    # elif act_index == 4:
    #     move_type = 3
    #     updown = 1
    #     sign = 1 # doesn't matter
    #     return move_type, updown, sign
    
    else:
        raise Exception('Act_index can only have integers from 0 to 5 as its values.')

def choose_action(proba):
    act_num = Categorical(proba).sample()
    
    node_index = torch.div(act_num, 5, rounding_mode='floor').item()
    act_index = torch.remainder(act_num, 5).item()
    move_type, updown, sign = action_dict(act_index)
       
    return [node_index, move_type, updown, sign, act_num]

def save_graph(pyg_graph, timestep, name_part):
        # First convert a PyG graph to a `networkx` graph, then plot the graph by using `nx.draw` function.
        # Config 
        plt.figure(figsize=(7,7))
        plt.xticks([])
        plt.yticks([])

        g = to_networkx(pyg_graph, to_undirected=True) #convert to networkx graph

        labeldict = {} # node labels in the plot
        for i in range(pyg_graph.num_nodes):
            labeldict[i]=pyg_graph.x[i].item()
        # Plot
        pos_nodes = nx.kamada_kawai_layout(g)
        pos_attrs = {}
        for node, coords in pos_nodes.items():
            pos_attrs[node] = (coords[0], coords[1] + 0.08)
        nx.draw(g, pos_nodes, node_color='black', node_size=30)
        nx.draw_networkx_labels(g, pos_attrs, labels=labeldict)
        name = name_part + '_' + str(timestep) + ".png"
        plt.savefig(name, bbox_inches = 'tight')
        plt.close()
        
import networkx.algorithms.isomorphism as iso
def is_same(g1, g2):
    g1 = to_networkx(g1, to_undirected=True, node_attrs=["x"])
    g2 = to_networkx(g2, to_undirected=True, node_attrs=["x"])
    nm = iso.numerical_node_match("x", None)
    return nx.is_isomorphic(g1, g2, node_match=nm)
