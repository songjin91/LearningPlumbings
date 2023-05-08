import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
import random

from models import Actor, Critic
from utils import *

# initialize two models
model_a = Actor()
model_c = Critic()

# load from saved models
print("Loading Trained Models...")
model_a.load_state_dict(torch.load('model_a.pth'))
model_c.load_state_dict(torch.load('model_c.pth'))


# check performance
print("*****CHECK PERFORMANCE*****")
def change_repn(graph, n_moves):
    state = graph
    for _ in range(n_moves):
        node_index = np.random.randint(len(state.x))
        move_type = np.random.randint(1, 4)
        updown = random.choice([1, -1])
        sign = random.choice([1, -1])
        _, state = neumann_move(state, node_index, move_type, updown, sign)
    return state

def get_next_state(state):
    # Get action from the current policy
    action_proba, _ = model_a(state)

    # Randomly sample an action from the probability distribution
    action = choose_action(action_proba)

    # Take action on the environment and get reward, next_state
    if action[2] == 1: # blow-up
        move_done, state = neumann_move(state, node_index=action[0], type=action[1], 
                            updown=action[2], sign=action[3])
    else: # blow-down
        for k in range(3):
            move_done, state = neumann_move(state, node_index=action[0], type=k+1, 
                            updown=action[2], sign=action[3])
            if move_done:
                break
    return state

def compare_two_graphs(state_1, state_2):
    done = False
    timestep = 0
    min_states_1 = []
    min_states_2 = []
    
    while not done:
        timestep += 1
        if timestep > 500:
            break
        state_1 = get_next_state(state_1)
        state_2 = get_next_state(state_2)

        if len(state_1.x) <= 10:
            min_states_1.append(state_1)
        if len(state_2.x) <= 10:
            min_states_2.append(state_1)

        if len(min_states_1) > 0 and len(min_states_2) > 0:
            for idx_1 in range(len(min_states_1)):
                for idx_2 in range(len(min_states_2)):
                    if is_same(min_states_1[idx_1], min_states_2[idx_2]):
                        done = True
    
    return done, timestep

n_success = 0
times_suc = 0  
 
for example_idx in range(10000):
    state = random_graph(n_max=10)
    state_1 = change_repn(state, 100)
    state_2 = change_repn(state, 100)
    done, tstep = compare_two_graphs(state_1, state_2)
    if done:
        n_success += 1
        times_suc += tstep
    if (example_idx+1) % 1000 == 0:

        print(f"Processing {example_idx+1} of 10000:")
        print(f"Number of SUCCESS: {n_success}")

acc = n_success/10000


print(f"TOTAL:    {n_success}")
print(f"ACCURACY: {acc}")
print(f"Average Moves: {times_suc/10000}")


