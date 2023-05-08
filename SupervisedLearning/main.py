import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
import random
from random import shuffle
from utils import *
from models import *

seed = 1234
random.seed(seed)
np.random.seed(seed + 1)
torch.manual_seed(seed + 2)

# Generate training datasets
dataset1 = generate_dataset(n_pairs = 10000, n_moves = 41, status = 'same', g1_move = True, g2_move = True)
dataset2 = generate_dataset(n_pairs = 20000, n_moves = 41, status = 'random', g1_move = True, g2_move = True)
dataset3 = generate_dataset(n_pairs = 20000, n_moves = 41, status = 'random', g1_move = True, g2_move = False)
dataset4 = generate_dataset(n_pairs = 20000, n_moves = 41, status = 'random', g1_move = False, g2_move = True)
dataset5 = generate_dataset_tweak(n_pairs = 10000)

dataset = dataset1 + dataset2 + dataset3 + dataset4 + dataset5
data_size = len(dataset)
idx = list(range(data_size))
shuffle(idx)
train_idx = idx[:int(data_size * 0.8)]
test_idx = idx[int(data_size * 0.8):]
train_set = [dataset[index] for index in train_idx]
test_set = [dataset[index] for index in test_idx]
train_loader = DataLoader(train_set, batch_size = 128, shuffle=True, follow_batch=['x_s', 'x_t'])
test_loader = DataLoader(test_set, batch_size = 128, shuffle=True, follow_batch=['x_s', 'x_t'])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.manual_seed(seed + 3)
print(f'Device Available: {device}')


# Initialize the models
model_1 = GENGAT().to(device)
model_2 = GCNGAT().to(device)
model_3 = GCNGCN().to(device)

# Start Training
print('##### Training session #####')

criterion = torch.nn.CrossEntropyLoss() # Loss function

def train(model):
    running_loss = 0
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for data in train_loader:        # Iterate in batches over the training dataset.
        data = data.to(device)
        optimizer.zero_grad()        # Clear gradients.
        x= model(data)               # Perform a single forward pass.
        loss = criterion(x, data.y)  # Compute the loss.
        running_loss += loss.item()
        loss.backward()              # Derive gradients.
        optimizer.step()             # Update parameters based on gradients.
    return running_loss/len(train_loader)

def test(loader, model):
    model.eval()
    correct = 0

    for data in loader:                         # Iterate in batches over the training/test dataset.
        data = data.to(device)
        out = model(data)  
        pred = out.argmax(dim=1)                # Use the class with highest probability.
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
    return correct / len(loader.dataset)        # Derive ratio of correct predictions.

def run_epsiodes(model, ep_max, name):
    train_acc = []
    test_acc = []
    loss_buffer = []
    test_max = 0
    for epoch in range(0, ep_max):
        loss_buffer.append(train(model))
        trainx = test(train_loader, model)
        testx = test(test_loader, model)
        
        if testx > test_max:
            test_max = testx
            torch.save(model.state_dict(), name+'.pth')
            print(f"****Model saved at epoch{epoch}")
            
        train_acc.append(trainx)
        test_acc.append(testx)
        if epoch % 10 == 0:           
            print(f'Epoch: {epoch:04d}, Train Acc: {trainx:.4f}, Test Acc: {testx:.4f}')

    print('===========================================')
    print('Train ACC:')
    print(f'    Minimum {np.min(train_acc) : .4f} \n    Maximum {np.max(train_acc) : .4f} ')
    print('Test ACC:')
    print(f'    Minimum {np.min(test_acc) : .4f} \n    Maximum {np.max(test_acc) : .4f} ')
    return train_acc, test_acc, loss_buffer
    
episodes_max = 150
learning_rate = 0.001

print('Training GENGAT Model...')
train_acc_1, test_acc_1, loss_buffer_1 = run_epsiodes(model_1, episodes_max, "GENGAT")
print('Training GENGAT Model...')
train_acc_2, test_acc_2, loss_buffer_2 = run_epsiodes(model_2, episodes_max, "GCNGAT")
print('Training GENGAT Model...')
train_acc_3, test_acc_3, loss_buffer_3 = run_epsiodes(model_3, episodes_max, "GCNGCN")

# Visualize Training Results
# Plot Loss
epoch = range(1, len(loss_buffer_1)+1)
plt.plot(epoch, loss_buffer_1, color='r', label='GEN+GAT')
plt.plot(epoch, loss_buffer_2, color='g', label='GCN+GAT')
plt.plot(epoch, loss_buffer_3, color='b', label='GCN+GCN')
plt.legend(loc='best')
plt.savefig('img_loss.png', bbox_inches = 'tight')

# Plot Accuracy
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 5))
fig.suptitle('Accuracy vs Epochs')

epoch = range(1, len(train_acc_1)+1)
ax1.plot(epoch, train_acc_1, color='r', label='GEN+GAT')
ax1.plot(epoch, train_acc_2, color='g', label='GCN+GAT')
ax1.plot(epoch, train_acc_3, color='b', label='GCN+GCN')
ax1.set_xlabel('Epoch') 
ax1.set_ylabel('Accuracy')
ax1.set_title("Train Accuracy Curve") 

ax2.plot(epoch, test_acc_1, color='r', label='GEN+GAT')
ax2.plot(epoch, test_acc_2, color='g', label='GCN+GAT')
ax2.plot(epoch, test_acc_3, color='b', label='GCN+GCN')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy') 
ax2.set_title("Validation Accuracy Curve") 
plt.legend(loc='best')
plt.savefig('img_accuracy.png', bbox_inches = 'tight')


print('##### Finished Training Session #####')
print('===========================================')

# Save Data
torch.save(loss_buffer_1, 'loss_1.pt')
torch.save(loss_buffer_2, 'loss_2.pt')
torch.save(loss_buffer_3, 'loss_3.pt')

torch.save(train_acc_1, 'train_acc_1.pt')
torch.save(train_acc_2, 'train_acc_2.pt')
torch.save(train_acc_3, 'train_acc_3.pt')

torch.save(test_acc_1, 'test_acc_1.pt')
torch.save(test_acc_2, 'test_acc_2.pt')
torch.save(test_acc_3, 'test_acc_3.pt')

# Test Session
print('Start Test Session')

# Load the saved models
model_1 = GENGAT()
model_1.load_state_dict(torch.load('GENGAT.pth'))

model_2 = GCNGAT()
model_2.load_state_dict(torch.load('GCNGAT.pth'))

model_3 = GCNGCN()
model_3.load_state_dict(torch.load('GCNGCN.pth'))

print('Models Loaded.')
print('Preparing datasets for the tests...')

test_dataset_1 = generate_dataset(n_pairs = 10000, n_moves = 41, status = 'random')
test_loader_1 = DataLoader(test_dataset_1, batch_size=128, shuffle=True, follow_batch=['x_s', 'x_t'])

test_dataset_2 = generate_dataset(n_pairs = 10000, n_moves = 61, status = 'random')
test_loader_2 = DataLoader(test_dataset_2, batch_size=128, shuffle=True, follow_batch=['x_s', 'x_t'])

test_dataset_3 = generate_dataset(n_pairs = 10000, n_moves = 81, status = 'random')
test_loader_3 = DataLoader(test_dataset_3, batch_size=128, shuffle=True, follow_batch=['x_s', 'x_t'])

# Create a dataset of pairs of graphs with same determinants.
sd_list = []
x = np.array([1, 2, 5, 3])
x = np.reshape(x, (-1, 1))
a = np.array([[0, 1, 1, 1,], [1, 0, 0, 0,], [1, 0, 0, 0,], [1, 0, 0, 0,]])
sd_list.append(create_graph(x, a))

x = np.array([1, 2, 7, 3])
x = np.reshape(x, (-1, 1))
a = np.array([[0, 1, 1, 1,], [1, 0, 0, 0,], [1, 0, 0, 0,], [1, 0, 0, 0,]])
sd_list.append(create_graph(x, a))

x = np.array([2, 1, 1, 1])
x = np.reshape(x, (-1, 1))
a = np.array([[0, 1, 1, 1,], [1, 0, 0, 0,], [1, 0, 0, 0,], [1, 0, 0, 0,]])
sd_list.append(create_graph(x, a))

x = np.array([1, 1, -3, 2, -3, 7])
x = np.reshape(x, (-1, 1))
a = np.array([[0,1,1,1,0,0], [1,0,0,0,1,1], [1,0,0,0,0,0], [1, 0, 0, 0,0,0], [0,1,0,0,0,0], [0,1,0,0,0,0]])
sd_list.append(create_graph(x, a))

dataset_sdet = []
for i in range(len(sd_list)):
    for j in range(len(sd_list)):
        g1 = sd_list[i]
        for k in range(np.random.randint(1, 16)):
            g1 = neumann_move(g1)
        g2 = sd_list[j]
        for k in range(np.random.randint(1, 16)):
            g2 = neumann_move(g2)
        if i == j:
            y = 1
        else:
            y = 0
            
        g_pair = PairData(edge_index_s = g1.edge_index, x_s = g1.x, edge_index_t = g2.edge_index, x_t = g2.x, y = y)
        dataset_sdet.append(g_pair)

test_loader_4 = DataLoader(dataset_sdet, batch_size=16, shuffle=True, follow_batch=['x_s', 'x_t'])


print('#### For GENGAT model')
print(test(test_loader_1, model_1))
print(test(test_loader_2, model_1))
print(test(test_loader_3, model_1))
print(test(test_loader_4, model_1))

print('#### For GCNGAT model')
print(test(test_loader_1, model_2))
print(test(test_loader_2, model_2))
print(test(test_loader_3, model_2))
print(test(test_loader_4, model_2))

print('#### For GCNGCN model')
print(test(test_loader_1, model_3))
print(test(test_loader_2, model_3))
print(test(test_loader_3, model_3))
print(test(test_loader_4, model_3))
